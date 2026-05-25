import unittest
import numpy as np

from netochi.mapping.three_step_mapping.clustering.clusterer.utils_from_hierarchical_community_detection_paper.cluster import \
    Hierarchy, Partition
from netochi.mapping.three_step_mapping.clustering.clusterer.hcd_clusterer import HcdClusterer


class TestHierarchicalOutput(unittest.TestCase):

    def test_convert_to_hierarchical_output(self):
        level0 = [0, 0, 1, 1]  # node 0,1 -> cluster 0; node 2,3 -> cluster 1
        hierarchy = Hierarchy(partition=Partition(pvec=level0))
        level1 = [0, 0]  # 2 clusters merge into 1 root
        hierarchy.add_level(Partition(pvec=level1))

        instance = HcdClusterer()
        output = instance._convert_to_hierarchical_output(hierarchy)

        # Verify labels (Index = Node ID, Value = Cluster ID from level 0)
        expected_labels = np.array([0, 0, 1, 1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        # Verify cluster_parent mapping (Index = Cluster ID, Value = Parent ID)
        expected_parents = np.array([2, 2, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)

    def test_convert_to_hierarchical_output_three_level_hierarchy(self):
        # Level 0: 4 nodes, 4 clusters (Identity)
        level0 = [0, 1, 2, 3]
        hierarchy = Hierarchy(partition=Partition(pvec=level0))

        # Level 1: 4 clusters merge into 2 parents (0&1 -> 0, 2&3 -> 1)
        level1 = [0, 0, 1, 1]
        hierarchy.add_level(Partition(pvec=level1))

        # Level 2: 2 clusters merge into 1 root (0&1 -> 0)
        level2 = [0, 0]
        hierarchy.add_level(Partition(pvec=level2))

        instance = HcdClusterer()
        output = instance._convert_to_hierarchical_output(hierarchy)

        # 1. Labels (Node -> Cluster Level 0)
        expected_labels = np.array([0, 1, 2, 3], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        # 2. Hierarchy Map
        # Index 0..3: Leaf Clusters -> Parents 4 and 5
        # Index 4..5: Routers -> Root 6
        # Index 6: Root node -> -1
        expected_parents = np.array([4, 4, 5, 5, 6, 6, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)

        self.assertEqual(output.cluster_parent[0], 4)
        self.assertEqual(output.num_clusters, 4)

    def test_convert_to_hierarchical_output_single_cluster_pass_through(self):
        # Level 0: 2 nodes into 1 cluster
        level0 = [0, 0]
        hierarchy = Hierarchy(partition=Partition(pvec=level0))

        # Level 1: 1 cluster stays 1 cluster (identity mapping)
        level1 = [0]
        hierarchy.add_level(Partition(pvec=level1))

        instance = HcdClusterer()
        output = instance._convert_to_hierarchical_output(hierarchy)

        # Labels
        expected_labels = np.array([0, 0], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        # Parent (Child Cluster 0 points to Parent 1, which points to -1)
        expected_parents = np.array([1, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)
        self.assertEqual(output.num_clusters, 1)

    def test_convert_to_hierarchical_output_incremental_hierarchy_mapping(self):
        # Level 0: 4 nodes, 3 clusters (0, 1, 2)
        level0_pvec = [0, 0, 1, 2]
        hierarchy = Hierarchy(partition=Partition(pvec=level0_pvec))

        # Level 1: 3 clusters mapping to 2 parents
        # Mapping: Cluster 0 -> Parent 0 | Cluster 1 -> Parent 1 | Cluster 2 -> Parent 1
        level1_pvec = [0, 1, 1]
        hierarchy.add_level(Partition(pvec=level1_pvec))

        # Level 2: 2 clusters mapping to 1 root
        # Mapping: Child 0 -> Root 0 | Child 1 -> Root 0
        level2_pvec = [0, 0]
        hierarchy.add_level(Partition(pvec=level2_pvec))

        instance = HcdClusterer()
        output = instance._convert_to_hierarchical_output(hierarchy)

        # --- Verification ---

        # 1. Labels (from Level 0)
        expected_labels = np.array([0, 0, 1, 2], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        # 2. Parents
        # Cluster 0 -> 3
        # Clusters 1, 2 -> 4
        # Routers 3, 4 -> 5
        # Root 5 -> -1
        expected_parents = np.array([3, 4, 4, 5, 5, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)
        self.assertEqual(output.num_clusters, 3)
