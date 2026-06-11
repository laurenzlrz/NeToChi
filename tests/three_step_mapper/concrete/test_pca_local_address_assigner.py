import unittest
import numpy as np
import graph_tool.all as gt

from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput
from netochi.mapping.three_step_mapping.local_address_assignment.pca_local_address_assigner import PcaLocalAddressAssigner




class TestPcaLocalAddressAssigner(unittest.TestCase):

    def setUp(self):
        self.assigner = PcaLocalAddressAssigner()

    def test_single_node_per_core(self):
        """Verify that cores containing only 1 node always get assigned local index 0."""
        g = gt.Graph(directed=False)
        g.add_vertex(3)  # 3 isolated nodes

        # Each node is isolated in its own dedicated core (0, 1, 2)
        clustering = ClusterAndHwOutput(
            cluster_assignment=np.array([0, 1, 2], dtype=np.int_), num_clusters=3, hw=None
        )

        result = self.assigner.assign_addresses(g, clustering)

        # Every node should be index 0 inside its respective core
        expected = np.array([0, 0, 0], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)

    def test_identical_connectivity_fallback(self):
        """Verify fallback mechanism when all nodes in a core have identical connectivity profiles."""
        g = gt.Graph(directed=False)
        g.add_vertex(4)

        # All 4 nodes are placed into the exact same core
        clustering = ClusterAndHwOutput(
            cluster_assignment=np.array([0, 0, 0, 0], dtype=np.int_), num_clusters=1, hw=None
        )

        # Because the graph is completely empty/disconnected, all rows in the adjacency
        # matrix are uniformly zero. The fallback should trigger sequential assignment.
        result = self.assigner.assign_addresses(g, clustering)

        expected = np.array([0, 1, 2, 3], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)

    def test_pca_sorting_determinism(self):
        """Verify that distinct connectivity structures are sorted and grouped uniquely."""
        g = gt.Graph(directed=False)
        g.add_vertex(3)

        # All nodes are in core 0, but we give them distinct connectivity gradients
        clustering = ClusterAndHwOutput(
            cluster_assignment=np.array([0, 0, 0], dtype=np.int_), num_clusters=1, hw=None
        )

        # Node 0 has no connections
        # Node 1 connects to Node 2
        g.add_edge(1, 2)

        result = self.assigner.assign_addresses(g, clustering)

        # Ensure that every node got a unique, valid local address slot [0, 1, 2]
        self.assertEqual(set(result), {0, 1, 2})
        self.assertEqual(len(result), 3)


    def test_multi_core_heterogeneous_boundaries(self):
        """Verify address isolation across multiple cores running mixed size groups."""
        g = gt.Graph(directed=False)
        g.add_vertex(5)

        # Layout:
        # Core 0: Nodes 0, 1, 2
        # Core 1: Nodes 3, 4
        cluster_assignment = np.array([0, 0, 0, 1, 1], dtype=np.int_)
        clustering = ClusterAndHwOutput(cluster_assignment=cluster_assignment, num_clusters=2, hw=None)

        # Add a few edges to differentiate profiles inside Core 0
        g.add_edge(0, 1)

        result = self.assigner.assign_addresses(g, clustering)

        # Extract indices belonging to Core 0 and Core 1
        core_0_addresses = result[0:3]
        core_1_addresses = result[3:5]

        # Assert local indices are tightly bound to their relative sub-blocks
        self.assertEqual(set(core_0_addresses), {0, 1, 2})
        self.assertEqual(set(core_1_addresses), {0, 1})


    def test_empty_or_skipped_core_ids(self):
        """Ensure code is robust against skipped core ID values in the assignment sequence."""
        g = gt.Graph(directed=False)
        g.add_vertex(2)

        # Core IDs skip ID '1' entirely (0 and 2 used)
        cluster_assignment = np.array([0, 2], dtype=np.int_)
        clustering = ClusterAndHwOutput(cluster_assignment=cluster_assignment, num_clusters=2, hw=None)

        # Should handle skipped partitions elegantly without crashing
        result = self.assigner.assign_addresses(g, clustering)

        expected = np.array([0, 0], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)

