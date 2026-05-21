import unittest

from netochi.mapping.three_step_mapping.clustering.cluster_adapter.flatten_and_rebuild_adapter import \
    FlatRebuildClusteringAdapter
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput


class TestFlatRebuildClusteringAdapter(unittest.TestCase):

    def test_standard_balanced_tree(self):
        """Test a clean, perfectly balanced tree layout."""

        clustering = HierarchicalClusterOutput(
            cluster_assignment={0: 0, 1: 0, 2: 1, 3: 2, 4: 3},
            num_clusters=4,
            cluster_parent={
                0: 4,  # Leaf 0's parent is Router 4
                1: 4,  # Leaf 1's parent is Router 4
                2: 5,  # Leaf 2's parent is Router 5
                3: 5,  # Leaf 3's parent is Router 5
                4: 6,  # Router 4's parent is Root 6
                5: 6,  # Router 5's parent is Root 6
                6: -1  # Root node
            }
        )

        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)

        # log_2(4) = 2 levels
        self.assertEqual(result.hw.nodes_per_router, 2)
        self.assertEqual(result.hw.neurons_per_core, 2)
        self.assertEqual(result.hw.router_levels, 2)
        self.assertEqual(result.num_clusters, 4)  # 2^2 = 4
        self.assertEqual(result.cluster_assignment, clustering.cluster_assignment)

    def test_fractional_mean_rounding(self):
        """Test that mean children counts round up correctly to integers."""

        clustering = HierarchicalClusterOutput(
            cluster_assignment={0: 0, 1: 1},
            num_clusters=5,
            cluster_parent={
                0: 5, 1: 5, 2: 5,  # Router 5 has 3 children
                3: 6, 4: 6,  # Router 6 has 2 children
                5: 7, 6: 7,  # Root 7 has 2 children
                7: -1
            }
        )

        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)

        # math.ceil(2.5) = 3
        self.assertEqual(result.hw.nodes_per_router, 3)
        # log_3(5) = 1.46 -> ceil to 2 levels
        self.assertEqual(result.hw.router_levels, 2)
        self.assertEqual(result.num_clusters, 9)  # 3^2 = 9 (Includes 4 dummy slots)

    def test_single_cluster_edge_case(self):
        """Test tree with only 1 cluster requires 0 router levels."""

        clustering = HierarchicalClusterOutput(
            cluster_assignment={0: 0, 1: 0},
            num_clusters=1,
            cluster_parent={
                0: -1  # Leaf 0 is the root itself
            }
        )

        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)

        self.assertEqual(result.hw.router_levels, 0)
        self.assertEqual(result.num_clusters, 1)  # 1^0 = 1

    def test_single_child_per_router_safety(self):
        """Verify the log base 1 protection works if mean children count drops to 1."""

        clustering = HierarchicalClusterOutput(
            cluster_assignment={0: 0, 1: 0},
            num_clusters=1,
            cluster_parent={
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: -1
            }
        )

        # Should not raise ZeroDivisionError
        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)
        self.assertEqual(result.hw.nodes_per_router, 1)
        self.assertEqual(result.hw.router_levels, 0)


if __name__ == '__main__':

    unittest.main()