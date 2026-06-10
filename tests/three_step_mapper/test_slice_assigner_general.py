import unittest
import numpy as np
import graph_tool.all as gt

from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput, MosaicHardwareConfig
from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner




class TestOptimalSliceAssigner(unittest.TestCase):

    def setUp(self):
        self.assigner = DeltaOptimalSliceAssigner()

    def test_selects_first_half_slice(self):
        """
        Scenario: Target neuron 0 listens to the first half of neurons in Cluster 1.
        Hardware: 1 router level, max distance = 1, 4 neurons per core, slice factor = 2.
        At distance 1, the core is divided into 2 slices: [0, 2) and [2, 4).
        """
        hw = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=4,
            router_levels=1,
            slice_factor=2
        )

        # Node 0 -> Core 0 | Nodes 1, 2, 3, 4 -> Core 1
        cluster_assignment = np.array([0, 1, 1, 1, 1], dtype=np.int_)
        clustering = ClusterAndHwOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=2,
            hw=hw
        )

        g = gt.Graph(directed=True)
        g.add_vertex(5)

        # Target node 0 listens to nodes 1 and 2
        g.add_edge(1, 0)
        g.add_edge(2, 0)

        # Local addresses in Core 1:
        # Node 1 -> 0, Node 2 -> 1 (Both fall in Slice 0: local addresses [0, 2))
        # Node 3 -> 2, Node 4 -> 3
        local_assignment = np.array([0, 0, 1, 2, 3], dtype=np.int_)

        result = self.assigner.assign_slices(clustering, g, local_assignment)

        # Ensure correct shape: (5 vertices, max_distance 1 + 1) -> (5, 2)
        self.assertEqual(result.shape, (5, 2))

        # Target node 0 at distance 1 should select Slice 0
        self.assertEqual(result[0][1], 0)

    def test_selects_second_half_slice(self):
        """
        Scenario: Target neuron 0 listens to the second half of neurons in Cluster 1.
        Expected: Selects Slice 1 (local addresses [2, 4)).
        """
        hw = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=4,
            router_levels=1,
            slice_factor=2
        )

        cluster_assignment = np.array([0, 1, 1, 1, 1], dtype=np.int_)
        clustering = ClusterAndHwOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=2,
            hw=hw
        )

        g = gt.Graph(directed=True)
        g.add_vertex(5)

        # Target node 0 listens to nodes 3 and 4 (the second half)
        g.add_edge(3, 0)
        g.add_edge(4, 0)

        local_assignment = np.array([0, 0, 1, 2, 3], dtype=np.int_)

        result = self.assigner.assign_slices(clustering, g, local_assignment)

        # Target node 0 at distance 1 should select Slice 1
        self.assertEqual(result[0][1], 1)

    def test_multi_distance_isolation(self):
        """
        Scenario: Hardware with 2 router levels (max distance = 2).
        Target neuron 0 (Core 0) has a distance 1 source and a distance 2 source.
        Ensures slices are independently optimized per hierarchical distance level.
        """
        hw = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=4,
            router_levels=2,  # Max distance is now 2
            slice_factor=2
        )

        # According to `core_distance` logic:
        # dist(Core 0, Core 1) = 1 (mismatch at level 0)
        # dist(Core 0, Core 2) = 2 (mismatch at level 1)
        cluster_assignment = np.array([0, 1, 2], dtype=np.int_)
        clustering = ClusterAndHwOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=4,  # 2^2 total cores
            hw=hw
        )

        g = gt.Graph(directed=True)
        g.add_vertex(3)
        g.add_edge(1, 0)  # Source at distance 1
        g.add_edge(2, 0)  # Source at distance 2

        # Node 1 (Dist 1) -> local addr 3.
        # Slices at Dist 1 (2 slices): Slice 1 handles [2, 4).
        # Node 2 (Dist 2) -> local addr 2.
        # Slices at Dist 2 (4 slices): Slice 2 handles [2, 3).
        local_assignment = np.array([0, 3, 2], dtype=np.int_)

        result = self.assigner.assign_slices(clustering, g, local_assignment)

        # Verify correct distance allocations
        self.assertEqual(result[0][1], 1)  # Slice 1 for distance 1
        self.assertEqual(result[0][2], 2)  # Slice 2 for distance 2

    def test_tie_break_with_no_inputs(self):
        """
        Scenario: Target neuron has no incoming connections.
        It should safely default to slice 0.
        """
        hw = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=4,
            router_levels=1,
            slice_factor=2
        )

        cluster_assignment = np.array([0, 1], dtype=np.int_)
        clustering = ClusterAndHwOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=2,
            hw=hw
        )

        g = gt.Graph(directed=True)
        g.add_vertex(2)
        # No edges added -> 0 incoming connections

        local_assignment = np.array([0, 0], dtype=np.int_)

        result = self.assigner.assign_slices(clustering, g, local_assignment)

        self.assertEqual(result[0][1], 0)

