
import unittest
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
        parent_map = {0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 6: -1}
        assignment_map = {10: 0, 11:1, 12: 2, 13: 3}  # 1 neuron per core

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=4,
            cluster_parent=parent_map
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        # Hardware Verification
        self.assertEqual(output.hw.nodes_per_router, 2)
        self.assertEqual(output.hw.router_levels, 2)
        self.assertEqual(output.hw.neurons_per_core, 1)
        self.assertEqual(output.num_clusters, 4)  # 2^2 = 4

        # Base-2 Positioning Verification
        # 3 -> branch 0,0 -> ID 0
        # 4 -> branch 0,1 -> ID 1
        # 5 -> branch 1,0 -> ID 2
        # 6 -> branch 1,1 -> ID 3
        expected_assignments = {10: 0, 11: 1, 12: 2, 13: 3}
        self.assertEqual(output.cluster_assignment, expected_assignments)

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
        parent_map = {0: 3, 1: 3, 2: 4, 3: 5, 4: 5, 5: -1}
        assignment_map = {100: 0, 200: 1, 300: 2}

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=3,
            cluster_parent=parent_map
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        # Max children count anywhere is 2 (at root and router 1)
        self.assertEqual(output.hw.nodes_per_router, 2)
        self.assertEqual(output.num_clusters, 4)  # Max capacity padded to 2^2

        # Check that core 5 gets shifted over to account for the implicit dummy
        # 3 -> ID 0
        # 4 -> ID 1
        # 5 -> (hw_id of 2 is 1) -> 1 * 2 + 0 (first sorted child) = 2
        expected_assignments = {100: 0, 200: 1, 300: 2}
        self.assertEqual(output.cluster_assignment, expected_assignments)

    def test_asymmetric_tree_requiring_padding2(self):
        """
        Scenario: Max branching factor is 2, but one router is missing a child branch.
        Tree Layout:
                7 (Root)
             /   \    \
            4     5    6
           / \     \    \
          0   1     2   3    (Leaf Clusters)

        Router 2 is missing its first child slot. This means physical slot 4
        (base-2 path 1,0) will be empty dummy space, and core 5 moves to slot 5 (path 1,1).
        """
        parent_map = {0: 4, 1: 4, 2: 5, 3: 6, 4:7, 5: 7, 6: 7, 7: -1}
        assignment_map = {100: 0, 200: 1, 300: 2, 400: 3}

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=4,
            cluster_parent=parent_map
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        # Max children count anywhere is 2 (at root and router 1)
        self.assertEqual(output.hw.nodes_per_router, 3)
        self.assertEqual(output.num_clusters, 9)  # Max capacity padded to 2^2

        # Check that core 5 gets shifted over to account for the implicit dummy
        # 3 -> ID 0
        # 4 -> ID 1
        # 5 -> (hw_id of 2 is 1) -> 1 * 2 + 0 (first sorted child) = 2
        expected_assignments = {100: 0, 200: 1, 300: 3, 400: 6}
        self.assertEqual(output.cluster_assignment, expected_assignments)

    def test_dense_neurons_per_core(self):
        """
        Scenario: Verifies that multiple neurons sitting inside the same core
        all get transitioned to the same physical hardware coordinate together.
        """
        parent_map = {0: 2, 1: 2, 2: -1}
        assignment_map = {
            10: 0, 11: 0, 12: 0,  # Core 1 has 3 neurons
            20: 1                 # Core 2 has 1 neuron
        }

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=2,
            cluster_parent=parent_map
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        self.assertEqual(output.hw.neurons_per_core, 3)  # Max capacity tracked correctly
        self.assertEqual(output.cluster_assignment[10], 0)
        self.assertEqual(output.cluster_assignment[11], 0)
        self.assertEqual(output.cluster_assignment[12], 0)
        self.assertEqual(output.cluster_assignment[20], 1)

    def test_large_branching_factor(self):
        """
        Scenario: A wide single-level router structure (Base-4 arithmetic).
              4 (Root)
           /  |  |  \
          0   1   2  3    (Leaf Clusters)
        """
        parent_map = {0: 4, 1: 4, 2: 4, 3: 4, 4: -1}
        assignment_map = {10: 0, 20: 1, 30: 2, 40: 3}

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=4,
            cluster_parent=parent_map
        )

        output = PaddingClusteringAdapter().adapt_clustering(clustering)

        self.assertEqual(output.hw.nodes_per_router, 4)
        self.assertEqual(output.hw.router_levels, 1)
        self.assertEqual(output.num_clusters, 4)  # 4^1 = 4

        expected_assignments = {10: 0, 20: 1, 30: 2, 40: 3}
        self.assertEqual(output.cluster_assignment, expected_assignments)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)