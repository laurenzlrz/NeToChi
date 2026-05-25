import unittest
import numpy as np

from netochi.mapping.three_step_mapping.clustering.cluster_adapter.padding_adapter import PaddingClusteringAdapter
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput


class TestAdaptClustering(unittest.TestCase):

    def test_perfectly_balanced_tree(self):
        """
        Scenario: A symmetric binary tree where every router has exactly 2 children.
        No structural padding should change the alignment positions.
        Tree Layout:
              6 (Root)
             /   \
            4     5
           / \   / \
          0   1 2   3 (Leaf Clusters)
        """
        # Array index = Cluster ID, Value = Parent Cluster ID
        cluster_parent = np.array([4, 4, 5, 5, 6, 6, -1], dtype=np.int_)

        # Neurons re-indexed to 0, 1, 2, 3
        # Array index = Neuron ID, Value = Assigned Leaf Cluster
        cluster_assignment = np.array([0, 1, 2, 3], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=4,
            cluster_parent=cluster_parent
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        # Hardware Verification
        self.assertEqual(output.hw.nodes_per_router, 2)
        self.assertEqual(output.hw.router_levels, 2)
        self.assertEqual(output.hw.neurons_per_core, 1)
        self.assertEqual(output.num_clusters, 4)  # 2^2 = 4

        # Base-2 Positioning Verification
        expected_assignments = np.array([0, 1, 2, 3], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_assignments)

    def test_asymmetric_tree_requiring_padding(self):
        """
        Scenario: Max branching factor is 2, but one router is missing a child branch.
        Tree Layout:
              5 (Root)
             /   \
            3     4
           / \     \
          0   1     2 (Leaf Clusters)

        Router 2 is missing its first child slot. This means physical slot 4
        (base-2 path 1,0) will be empty dummy space, and core 5 moves to slot 5 (path 1,1).
        """
        cluster_parent = np.array([3, 3, 4, 5, 5, -1], dtype=np.int_)

        # Neurons re-indexed to 0, 1, 2
        cluster_assignment = np.array([0, 1, 2], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=3,
            cluster_parent=cluster_parent
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        # Max children count anywhere is 2 (at root and router 1)
        self.assertEqual(output.hw.nodes_per_router, 2)
        self.assertEqual(output.num_clusters, 4)  # Max capacity padded to 2^2

        expected_assignments = np.array([0, 1, 2], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_assignments)

    def test_asymmetric_tree_requiring_padding2(self):
        """
        Scenario: Max branching factor is 2, but one router is missing a child branch.
        Tree Layout:
                7 (Root)
             /   \    \
            4     5    6
           / \     \    \
          0   1     2   3    (Leaf Clusters)
        """
        cluster_parent = np.array([4, 4, 5, 6, 7, 7, 7, -1], dtype=np.int_)

        # Neurons re-indexed to 0, 1, 2, 3
        cluster_assignment = np.array([0, 1, 2, 3], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=4,
            cluster_parent=cluster_parent
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        # Max children count anywhere is 3 (at root)
        self.assertEqual(output.hw.nodes_per_router, 3)
        self.assertEqual(output.num_clusters, 9)  # Max capacity padded to 3^2

        # Check that cores get shifted over to account for implicit dummy slots
        expected_assignments = np.array([0, 1, 3, 6], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_assignments)

    def test_dense_neurons_per_core(self):
        """
        Scenario: Verifies that multiple neurons sitting inside the same core
        all get transitioned to the same physical hardware coordinate together.
        """
        cluster_parent = np.array([2, 2, -1], dtype=np.int_)

        # Neurons re-indexed to 0, 1, 2, 3
        # Core 0 has 3 neurons (0, 1, 2), Core 1 has 1 neuron (3)
        cluster_assignment = np.array([0, 0, 0, 1], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=2,
            cluster_parent=cluster_parent
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        self.assertEqual(output.hw.neurons_per_core, 3)  # Max capacity tracked correctly
        self.assertEqual(output.cluster_assignment[0], 0)
        self.assertEqual(output.cluster_assignment[1], 0)
        self.assertEqual(output.cluster_assignment[2], 0)
        self.assertEqual(output.cluster_assignment[3], 1)

    def test_large_branching_factor(self):
        """
        Scenario: A wide single-level router structure (Base-4 arithmetic).
              4 (Root)
           /  |  |  \
          0   1   2  3    (Leaf Clusters)
        """
        cluster_parent = np.array([4, 4, 4, 4, -1], dtype=np.int_)

        # Neurons re-indexed to 0, 1, 2, 3
        cluster_assignment = np.array([0, 1, 2, 3], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=4,
            cluster_parent=cluster_parent
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        self.assertEqual(output.hw.nodes_per_router, 4)
        self.assertEqual(output.hw.router_levels, 1)
        self.assertEqual(output.num_clusters, 4)  # 4^1 = 4

        expected_assignments = np.array([0, 1, 2, 3], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_assignments)

