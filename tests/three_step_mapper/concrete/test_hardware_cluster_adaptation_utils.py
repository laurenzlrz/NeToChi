import unittest

import numpy as np

from netochi.mapping.three_step_mapping.clustering.hardware_cluster_adaptation_utils import \
    compute_avg_non_leaf_children_count, transform_hierarchy_into_adjacency_list, compute_core_sizes, \
    compute_hierarchy_depth
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput


def compute_avg_children(cluster_parent: np.ndarray) -> float:
    """
    Computes the average number of children per non-leaf node.
    Returns 0.0 if there are no non-leaf nodes.
    """
    output = HierarchicalClusterOutput(cluster_assignment=None, num_clusters=None, cluster_parent=cluster_parent)
    return compute_avg_non_leaf_children_count(cluster_output=output)



class TestHwClusterAdapterUtils(unittest.TestCase):

    def test_compute_hierarchy_depth_standard(self):
        """Test depth calculation on a standard multi-level tree architecture."""
        # 0 -> 4 -> 6 -> -1 (Tree depth is 2 levels of routing)
        cluster_parent = np.array([4, 4, 5, 5, 6, 6, -1], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=np.array([0, 0, 1, 2, 3], dtype=np.int_),
            num_clusters=4,
            cluster_parent=cluster_parent
        )

        depth = compute_hierarchy_depth(clustering)
        self.assertEqual(depth, 2)

    def test_compute_hierarchy_depth_flat(self):
        """Test depth calculation on a completely flat architecture (0 routing levels)."""
        # Node 0 is the root itself
        cluster_parent = np.array([-1], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=np.array([0, 0, 0], dtype=np.int_),
            num_clusters=1,
            cluster_parent=cluster_parent
        )

        depth = compute_hierarchy_depth(clustering)
        self.assertEqual(depth, 0)

    def test_compute_core_sizes_mixed(self):
        """Test counting core sizes with an uneven distribution of nodes per cluster."""
        # Core 0: 3 nodes (indices 0, 1, 2)
        # Core 1: 1 node  (index 3)
        # Core 2: 2 nodes (indices 4, 5)
        cluster_assignment = np.array([0, 0, 0, 1, 2, 2], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=3,
            cluster_parent=np.array([3, 3, 3, -1], dtype=np.int_)
        )

        core_sizes = compute_core_sizes(clustering)

        expected_sizes = {0: 3, 1: 1, 2: 2}
        self.assertEqual(core_sizes, expected_sizes)

    def test_compute_core_sizes_single_large_core(self):
        """Test counting core sizes when all nodes belong to a single core."""
        cluster_assignment = np.array([4, 4, 4, 4, 4], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=1,
            cluster_parent=np.array([-1], dtype=np.int_)
        )

        core_sizes = compute_core_sizes(clustering)

        expected_sizes = {4: 5}
        self.assertEqual(core_sizes, expected_sizes)


    def test_average_children_calculation(self):
        # --- Case 1: Perfectly Balanced Binary Tree ---
        # Routers 4 and 5 have 2 children each. Root 6 has 2 children.
        # Total non-leaf nodes = 3. Total children counted = 6. Average = 6 / 3 = 2.0
        parent_balanced = np.array([4, 4, 5, 5, 6, 6, -1], dtype=np.int_)
        assert compute_avg_children(parent_balanced) == 2.0

        # --- Case 2: Unbalanced Tree (Fractional Mean) ---
        # Router 5 has 3 children (0, 1, 2)
        # Router 6 has 2 children (3, 4)
        # Root 7 has 2 children (5, 6)
        # Non-zero counts: [3, 2, 2]. Average = (3 + 2 + 2) / 3 = 7 / 3 ≈ 2.3333
        parent_unbalanced = np.array([5, 5, 5, 6, 6, 7, 7, -1], dtype=np.int_)
        expected_mean = (3 + 2 + 2) / 3
        assert np.isclose(compute_avg_children(parent_unbalanced), expected_mean)

        # --- Case 3: Degenerate Tree (Chain/Pipeline) ---
        # Node 0 -> 1 -> 2 -> 3 -> -1
        # Every non-leaf node (1, 2, 3) has exactly 1 child. Average = 1.0
        parent_chain = np.array([1, 2, 3, -1], dtype=np.int_)
        assert compute_avg_children(parent_chain) == 1.0

        # --- Case 4: Single Node / Edge Case ---
        # Only a root node exists, meaning there are 0 non-leaf nodes.
        parent_single = np.array([-1], dtype=np.int_)
        assert compute_avg_children(parent_single) == 0.0



    def test_transform_hierarchy_standard_hierarchy(self):
        """Test a standard 3-level tree hierarchy with multiple leaves and neurons."""
        # Setup a tree where:
        #   0 (Root) -> 1, 2 (Routers)
        #   1 -> 3, 4 (Leaf Clusters)
        #   2 -> 5 (Leaf Cluster)
        # Array index = Cluster ID, Value = Parent ID
        cluster_parent = np.array([-1, 0, 0, 1, 1, 2], dtype=np.int_)

        # Neurons re-indexed to 0, 1, 2, 3, 4
        # Array index = Neuron ID, Value = Leaf Cluster ID
        cluster_assignment = np.array([3, 3, 4, 5, 5], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=3,
            cluster_parent=cluster_parent
        )

        children, leaf_neurons = transform_hierarchy_into_adjacency_list(clustering)

        # Verify Adjacency List (Children Mapping)
        expected_children = {
            0: np.array([1, 2], dtype=np.int_),
            1: np.array([3, 4], dtype=np.int_),
            2: np.array([5], dtype=np.int_)
        }

        self.assertEqual(set(children.keys()), set(expected_children.keys()))
        for parent_id, expected_arr in expected_children.items():
            np.testing.assert_array_equal(children[parent_id], expected_arr)

        # Verify Leaf to Neuron Mapping
        expected_leaf_neurons = {
            3: np.array([0, 1], dtype=np.int_),
            4: np.array([2], dtype=np.int_),
            5: np.array([3, 4], dtype=np.int_)
        }

        self.assertEqual(set(leaf_neurons.keys()), set(expected_leaf_neurons.keys()))
        for leaf_id, expected_arr in expected_leaf_neurons.items():
            np.testing.assert_array_equal(leaf_neurons[leaf_id], expected_arr)

    def test_transform_hierarchy_sorting_determinism(self):
        """Ensure that child IDs and neuron IDs are sorted deterministically."""
        # Nodes 1 and 2 both have parent 0
        cluster_parent = np.array([-1, 0, 0], dtype=np.int_)

        # Neurons 0 and 1 both belong to cluster 2
        cluster_assignment = np.array([2, 2], dtype=np.int_)

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=1,
            cluster_parent=cluster_parent
        )

        children, leaf_neurons = transform_hierarchy_into_adjacency_list(clustering)

        # Root 0's children must be sorted [1, 2]
        np.testing.assert_array_equal(children[0], np.array([1, 2], dtype=np.int_))

        # Cluster 2's neurons must be sorted [0, 1]
        np.testing.assert_array_equal(leaf_neurons[2], np.array([0, 1], dtype=np.int_))

    def test_transform_hierarchy_single_node_tree(self):
        """Test edge case where the tree only consists of a root node."""
        cluster_parent = np.array([-1], dtype=np.int_)
        cluster_assignment = np.array([0], dtype=np.int_)  # Neuron 0 is in Cluster 0

        clustering = HierarchicalClusterOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=1,
            cluster_parent=cluster_parent
        )

        children, leaf_neurons = transform_hierarchy_into_adjacency_list(clustering)

        # A root with no children should result in an empty children dict (or no valid keys)
        # Filtering out keys that don't have children or ensuring the dict is empty:
        active_parents = {k: v for k, v in children.items() if len(v) > 0}
        self.assertEqual(active_parents, {})

        # The single cluster should still map to its arrayed neuron ID
        np.testing.assert_array_equal(leaf_neurons[0], np.array([0], dtype=np.int_))
