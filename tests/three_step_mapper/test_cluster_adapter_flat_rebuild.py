import unittest
import numpy as np

from netochi.mapping.three_step_mapping.clustering.cluster_adapter.flatten_and_rebuild_adapter import \
    FlatRebuildClusteringAdapter
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput


class TestFlatRebuildClusteringAdapter(unittest.TestCase):

    def test_standard_balanced_tree(self):
        """Test a clean, perfectly balanced tree layout."""

        # Original keys (0, 1, 2, 3, 4) mapped to values
        cluster_assignment = np.array([0, 0, 1, 2, 3], dtype=np.int_)

        # Original keys (0 through 6) mapped to values
        cluster_parent = np.array([4, 4, 5, 5, 6, 6, -1], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=4,
            cluster_parent=cluster_parent
        )

        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)

        # log_2(4) = 2 levels
        self.assertEqual(result.hw.nodes_per_router, 2)
        self.assertEqual(result.hw.neurons_per_core, 2)
        self.assertEqual(result.hw.router_levels, 2)
        self.assertEqual(result.num_clusters, 4)  # 2^2 = 4
        np.testing.assert_array_equal(result.cluster_assignment, clustering.cluster_assignment)

    def test_fractional_mean_rounding(self):
        """Test that mean children counts round up correctly to integers."""

        # Original keys (0, 1) mapped to values
        cluster_assignment = np.array([0, 1], dtype=np.int_)

        # Original keys (0 through 7) mapped to values
        cluster_parent = np.array([5, 5, 5, 6, 6, 7, 7, -1], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=5,
            cluster_parent=cluster_parent
        )

        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)

        # math.ceil(2.5) = 3
        self.assertEqual(result.hw.nodes_per_router, 3)
        # log_3(5) = 1.46 -> ceil to 2 levels
        self.assertEqual(result.hw.router_levels, 2)
        self.assertEqual(result.num_clusters, 9)  # 3^2 = 9 (Includes 4 dummy slots)

    def test_single_cluster_edge_case(self):
        """Test tree with only 1 cluster requires 0 router levels."""

        cluster_assignment = np.array([0, 0], dtype=np.int_)
        cluster_parent = np.array([-1], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=1,
            cluster_parent=cluster_parent
        )

        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)

        self.assertEqual(result.hw.router_levels, 0)
        self.assertEqual(result.num_clusters, 1)  # 1^0 = 1

    def test_single_child_per_router_safety(self):
        """Verify the log base 1 protection works if mean children count drops to 1."""

        cluster_assignment = np.array([0, 0], dtype=np.int_)
        cluster_parent = np.array([1, 2, 3, 4, -1], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=1,
            cluster_parent=cluster_parent
        )

        # Should not raise ZeroDivisionError
        result = FlatRebuildClusteringAdapter().adapt_clustering(clustering)
        self.assertEqual(result.hw.nodes_per_router, 1)
        self.assertEqual(result.hw.router_levels, 0)
