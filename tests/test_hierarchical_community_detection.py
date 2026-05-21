import unittest
from dataclasses import dataclass
from typing import Dict

from netochi.mapping.three_step_mapping.clustering.hierarchical_community_detection.utils_from_paper.cluster import Hierarchy, Partition
from netochi.mapping.three_step_mapping.clustering.hierarchical_community_detection.hcd_clusterer import HcdClusterer


# Assuming these are your structures based on the code snippet
@dataclass
class HierarchicalClusterOutput:
    labels: Dict[int, int]
    cluster_parent: Dict[int, int]


class TestHierarchicalOutput(unittest.TestCase):
    def test_convert_to_hierarchical_output(self):

        level0 = [0, 0, 1, 1]  # node 0,1 -> cluster 10; node 2,3 -> cluster 11
        hierarchy = Hierarchy(partition=Partition(pvec=level0))
        level1 = [0, 0]  # 2 clusters at this level
        hierarchy.add_level(Partition(pvec=level1))

        instance = HcdClusterer()
        output = instance._convert_to_hierarchical_output(hierarchy)

        # Verify labels (Node ID -> Cluster ID from level 0)
        expected_labels = {0: 0, 1: 0, 2: 1, 3: 1}
        self.assertEqual(output.cluster_assignment, expected_labels)

        # Verify cluster_parent mapping
        expected_parents = {0: 2, 1: 2, 2: -1}
        self.assertEqual(output.cluster_parent, expected_parents)


    def test_three_level_hierarchy(self):
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
        self.assertEqual(output.cluster_assignment, {0: 0, 1: 1, 2: 2, 3: 3})

        expected_parents = {
            0: 4, 1: 4, 2: 5, 3: 5,  # From Level 1 loop
            4: 6, 5: 6, 6: -1  # From Level 2 loop (Note: IDs might overwrite if offset is 0)
        }
        # Important: If your code doesn't increment cluster_offset, keys 0 and 1 will be overwritten.
        self.assertEqual(output.cluster_parent, expected_parents)
        self.assertEqual(output.cluster_parent[0], 4)
        self.assertEqual(output.num_clusters, 4)

    def test_single_cluster_pass_through(self):
        # Level 0: 2 nodes into 1 cluster
        level0 = [0, 0]
        hierarchy = Hierarchy(partition=Partition(pvec=level0))

        # Level 1: 1 cluster stays 1 cluster (identity mapping)
        level1 = [0]
        hierarchy.add_level(Partition(pvec=level1))

        instance = HcdClusterer()
        output = instance._convert_to_hierarchical_output(hierarchy)

        # Labels
        self.assertEqual(output.cluster_assignment, {0: 0, 1: 0})

        # Parent (nr_clusters at Level 1 is 1)
        # Child 0 -> 0 + 1 = 1
        expected_parents = {0: 1, 1: -1}
        self.assertEqual(output.cluster_parent, expected_parents)
        self.assertEqual(output.num_clusters, 1)

    def test_incremental_hierarchy_mapping(self):
        # Level 0: 4 nodes, 3 clusters (0, 1, 2)
        # pvec_expanded represents node -> cluster mapping
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
        expected_labels = {0: 0, 1: 0, 2: 1, 3: 2}
        self.assertEqual(output.cluster_assignment, expected_labels)


        expected_parents = {
            0: 3,
            1: 4,
            2: 4,
            3: 5,
            4: 5,
            5: -1
        }
        self.assertEqual(output.cluster_parent, expected_parents)
        self.assertEqual(output.num_clusters, 3)
