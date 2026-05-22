import unittest
import numpy as np
import graph_tool.all as gt

from netochi.mapping.three_step_mapping.clustering.clusterer.hsbm_clusterer import HsbmClusterer


class TestHsbmClusterer(unittest.TestCase):

    def setUp(self):
        self.instance = HsbmClusterer()

    def test_convert_to_hierarchical_output(self):
        g = gt.Graph(directed=False)
        g.add_vertex(4)

        # FIX: Cast the inner lists to np.ndarray
        hierarchy = [np.array([0, 0, 1, 1]), np.array([0, 0])]

        real_state = gt.NestedBlockState(g, bs=hierarchy)
        output = self.instance._convert_to_hierarchical_output(real_state)

        expected_labels = np.array([0, 0, 1, 1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        expected_parents = np.array([2, 2, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)
        self.assertEqual(output.num_clusters, 2)

    def test_convert_to_hierarchical_output_three_level_hierarchy(self):
        g = gt.Graph(directed=False)
        g.add_vertex(4)

        # FIX: Cast the inner lists to np.ndarray
        hierarchy = [np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]), np.array([0, 0])]
        real_state = gt.NestedBlockState(g, bs=hierarchy)

        output = self.instance._convert_to_hierarchical_output(real_state)

        expected_labels = np.array([0, 1, 2, 3], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        expected_parents = np.array([4, 4, 5, 5, 6, 6, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)
        self.assertEqual(output.cluster_parent[0], 4)
        self.assertEqual(output.num_clusters, 4)

    def test_convert_to_hierarchical_output_single_cluster_pass_through(self):
        g = gt.Graph(directed=False)
        g.add_vertex(2)

        # FIX: Cast the inner lists to np.ndarray
        hierarchy = [np.array([0, 0]), np.array([0])]
        real_state = gt.NestedBlockState(g, bs=hierarchy)

        output = self.instance._convert_to_hierarchical_output(real_state)

        expected_labels = np.array([0, 0], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        expected_parents = np.array([1, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)
        self.assertEqual(output.num_clusters, 1)

    def test_convert_to_hierarchical_output_incremental_hierarchy_mapping(self):
        g = gt.Graph(directed=False)
        g.add_vertex(4)

        # FIX: Cast the inner lists to np.ndarray
        hierarchy = [np.array([0, 0, 1, 2]), np.array([0, 1, 1]), np.array([0, 0])]
        real_state = gt.NestedBlockState(g, bs=hierarchy)

        output = self.instance._convert_to_hierarchical_output(real_state)

        expected_labels = np.array([0, 0, 1, 2], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_assignment, expected_labels)

        expected_parents = np.array([3, 4, 4, 5, 5, -1], dtype=np.int_)
        np.testing.assert_array_equal(output.cluster_parent, expected_parents)
        self.assertEqual(output.num_clusters, 3)

