import unittest

from netochi.mapping.three_step_mapping.clustering.hardware_cluster_adaptation_utils import transform_hierarchy_into_adjacency_list
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput


class TestTransformHierarchy(unittest.TestCase):

    def test_standard_hierarchy(self):
        """Test a standard 3-level tree hierarchy with multiple leaves and neurons."""
        # Setup a tree where:
        #   0 (Root) -> 1, 2 (Routers)
        #   1 -> 3, 4 (Leaf Clusters)
        #   2 -> 5 (Leaf Cluster)
        parent_map = {0: -1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}

        # Neurons mapped to leaf clusters 3, 4, and 5
        assignment_map = {
            100: 3, 101: 3,  # Cluster 3 has 2 neurons
            102: 4,  # Cluster 4 has 1 neuron
            103: 5, 104: 5  # Cluster 5 has 2 neurons
        }

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=3,
            cluster_parent=parent_map
        )

        children, leaf_neurons = transform_hierarchy_into_adjacency_list(clustering)

        # Verify Adjacency List (Children Mapping)
        expected_children = {
            0: [1, 2],
            1: [3, 4],
            2: [5]
        }
        self.assertEqual(children, expected_children)

        # Verify Leaf to Neuron Mapping
        expected_leaf_neurons = {
            3: [100, 101],
            4: [102],
            5: [103, 104]
        }
        self.assertEqual(leaf_neurons, expected_leaf_neurons)

    def test_sorting_determinism(self):
        """Ensure that child IDs and neuron IDs are sorted deterministically regardless of insertion order."""
        # Intentionally insert out of numerical order
        parent_map = {0: -1, 2: 0, 1: 0}
        assignment_map = {105: 2, 102: 2}

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=1,
            cluster_parent=parent_map
        )

        children, leaf_neurons = transform_hierarchy_into_adjacency_list(clustering)

        # Root 0's children should be sorted [1, 2], not [2, 1]
        self.assertEqual(children[0], [1, 2])

        # Cluster 2's neurons should be sorted [102, 105], not [105, 102]
        self.assertEqual(leaf_neurons[2], [102, 105])

    def test_single_node_tree(self):
        """Test edge case where the tree only consists of a root node (which is also the leaf)."""
        parent_map = {0: -1}
        assignment_map = {999: 0}

        clustering = HierarchicalClusterOutput(
            cluster_assignment=assignment_map,
            num_clusters=1,
            cluster_parent=parent_map
        )

        children, leaf_neurons = transform_hierarchy_into_adjacency_list(clustering)

        # A root with no children should result in an empty children dict
        self.assertEqual(children, {})

        # The single cluster should still map to its neuron
        self.assertEqual(leaf_neurons, {0: [999]})


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)