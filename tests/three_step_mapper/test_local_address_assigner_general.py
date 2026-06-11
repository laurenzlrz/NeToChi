import unittest
import numpy as np
import graph_tool.all as gt

from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput
from netochi.mapping.three_step_mapping.local_address_assignment.pca_local_address_assigner import \
    PcaLocalAddressAssigner


class TestSliceLocalAddressAssigner(unittest.TestCase):

    def setUp(self):
        self.assigner = PcaLocalAddressAssigner()

    def test_shared_downstream_target_locality(self):
        """
        User Scenario:
        Cluster A (Cores 0-3) and Cluster B (Cores 4-7).
        Neuron 0 in Cluster A receives input from Neurons 4 and 5 in Cluster B.
        Neurons 4 and 5 should be mapped to the same half of Cluster B
        (local addresses {0, 1} OR {2, 3}).
        """
        g = gt.Graph(directed=True)
        g.add_vertex(8)

        # Cluster assignment array: Index = Neuron ID, Value = Cluster ID
        # Neurons 0,1,2,3 -> Cluster 0 (A)
        # Neurons 4,5,6,7 -> Cluster 1 (B)
        cluster_assignment = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
        clustering = ClusterAndHwOutput(cluster_assignment=cluster_assignment, num_clusters=2, hw=None)

        # Establish directed edges: 4 -> 0 and 5 -> 0
        g.add_edge(4, 0)
        g.add_edge(5, 0)

        # Run the assignment algorithm
        local_addresses = self.assigner.assign_addresses(g, clustering)

        # Extract local coordinates for the two source neurons in Cluster B
        addr_4 = local_addresses[4]
        addr_5 = local_addresses[5]

        # Verify they share the same local address half-block
        # Either both are in the lower half [0, 1] or both in the upper half [2, 3]
        in_same_half = (addr_4 < 2 and addr_5 < 2) or (addr_4 >= 2 and addr_5 >= 2)
        self.assertTrue(
            in_same_half,
            f"Neurons 4 and 5 split across halves! Got addresses {addr_4} and {addr_5}"
        )

    def test_bipartite_functional_segregation(self):
        """
        Scenario: Max segregation test.
        Cluster B (Neurons 4, 5, 6, 7) sends inputs to Cluster A.
        Neurons 4 and 5 target Neuron 0.
        Neurons 6 and 7 target Neuron 1.
        The pairs should segregate cleanly into opposing halves of Cluster B.
        """
        g = gt.Graph(directed=True)
        g.add_vertex(8)

        cluster_assignment = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
        clustering = ClusterAndHwOutput(cluster_assignment=cluster_assignment, num_clusters=2, hw=None)

        # Group 1 targeting Neuron 0
        g.add_edge(4, 0)
        g.add_edge(5, 0)

        # Group 2 targeting Neuron 1
        g.add_edge(6, 1)
        g.add_edge(7, 1)

        local_addresses = self.assigner.assign_addresses(g, clustering)

        # Determine which halves the groups landed in
        group_1_halves = [local_addresses[4] // 2, local_addresses[5] // 2]
        group_2_halves = [local_addresses[6] // 2, local_addresses[7] // 2]

        # 1. Ensure members of the same group stay together
        self.assertEqual(group_1_halves[0], group_1_halves[1])
        self.assertEqual(group_2_halves[0], group_2_halves[1])

        # 2. Ensure Group 1 and Group 2 land in completely different halves
        self.assertNotEqual(group_1_halves[0], group_2_halves[0])

    def test_address_space_integrity(self):
        """
        Scenario: Regardless of connection profiles, ensure the local
        address spaces are strictly legal permutations (no duplicates, no gaps).
        """
        g = gt.Graph(directed=True)
        g.add_vertex(6)

        # Cluster 0 has 4 neurons, Cluster 1 has 2 neurons
        cluster_assignment = np.array([0, 0, 0, 0, 1, 1], dtype=np.int_)
        clustering = ClusterAndHwOutput(cluster_assignment=cluster_assignment, num_clusters=2, hw=None)

        # Randomized connectivity to stress test sorting
        g.add_edge(0, 4)
        g.add_edge(2, 5)
        g.add_edge(3, 4)

        local_addresses = self.assigner.assign_addresses(g, clustering)

        # Extract address layouts per cluster block
        cluster_0_slots = local_addresses[0:4]
        cluster_1_slots = local_addresses[4:6]

        # Validate that slots contain perfect sequential indices starting from 0
        self.assertEqual(set(cluster_0_slots), {0, 1, 2, 3})
        self.assertEqual(set(cluster_1_slots), {0, 1})

