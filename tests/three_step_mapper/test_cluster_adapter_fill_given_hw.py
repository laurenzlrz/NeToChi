import unittest
from unittest.mock import MagicMock

import numpy as np

from netochi.mapping.exceptions import HardwareConstraintError
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.clustering.cluster_adapter.fill_given_hw_adapter import FillGivenHwAdapter


class TestFillGivenHwAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = FillGivenHwAdapter()

    def test_adapt_clustering_perfect_fit(self):
        """
        Scenario: 4 neurons grouped evenly into 2 clusters.
        Hardware allows 2 neurons per core across 2 cores (total capacity = 4).
        Expectation: Cluster 0 maps entirely to Core 0, Cluster 1 maps entirely to Core 1.
        """
        # Concrete instance of HierarchicalClusterOutput
        clustering = HierarchicalClusterOutput(
            cluster_assignment=np.array([0, 0, 1, 1], dtype=np.int_),
            cluster_parent=np.array([-1], dtype=np.int_),  # Unused by this adapter
            num_clusters=2
        )

        mock_hw = MagicMock(spec=MosaicHardwareConfig)
        mock_hw.neurons_per_core = 2
        mock_hw.total_cores = 2

        result = self.adapter.adapt_clustering(clustering, mock_hw)

        # Verification
        expected_cores = np.array([0, 0, 1, 1], dtype=np.int_)
        np.testing.assert_array_equal(result.cluster_assignment, expected_cores)
        self.assertEqual(result.num_clusters, 2)  # Must match hw_config.total_cores
        self.assertEqual(result.hw, mock_hw)

    def test_adapt_clustering_with_spill_over(self):
        """
        Scenario: 4 neurons where Cluster 0 has 3 neurons, and Cluster 1 has 1 neuron.
        Hardware allows 2 neurons per core.
        Expectation: Cluster 0 spills over. Core 0 gets 2 nodes from cluster 0.
                     Core 1 gets the last node of cluster 0 AND the node from cluster 1.
        """
        clustering = HierarchicalClusterOutput(
            cluster_assignment=np.array([0, 0, 0, 1], dtype=np.int_),
            cluster_parent=np.array([-1], dtype=np.int_),
            num_clusters=2
        )

        mock_hw = MagicMock(spec=MosaicHardwareConfig)
        mock_hw.neurons_per_core = 2
        mock_hw.total_cores = 2

        result = self.adapter.adapt_clustering(clustering, mock_hw)

        # Verification:
        # Neuron 0, 1 -> Core 0
        # Neuron 2, 3 -> Core 1 (Neuron 2 spilled over from cluster 0)
        expected_cores = np.array([0, 0, 1, 1], dtype=np.int_)
        np.testing.assert_array_equal(result.cluster_assignment, expected_cores)

    def test_adapt_clustering_sorts_by_cluster_id(self):
        """
        Scenario: Cluster IDs are provided out of order in the assignment list.
        Expectation: The adapter processes cluster 0 first, meaning the physical
                     nodes belonging to cluster 0 will get filled into early cores.
        """
        clustering = HierarchicalClusterOutput(
            cluster_assignment=np.array([1, 1, 0, 0], dtype=np.int_),
            cluster_parent=np.array([-1], dtype=np.int_),
            num_clusters=2
        )

        mock_hw = MagicMock(spec=MosaicHardwareConfig)
        mock_hw.neurons_per_core = 2
        mock_hw.total_cores = 2


        result = self.adapter.adapt_clustering(clustering, mock_hw)

        # Verification:
        # Cluster 0 processed first -> Nodes 2 and 3 placed on Core 0
        # Cluster 1 processed second -> Nodes 0 and 1 placed on Core 1
        expected_cores = np.array([1, 1, 0, 0], dtype=np.int_)
        np.testing.assert_array_equal(result.cluster_assignment, expected_cores)

    def test_adapt_clustering_hardware_overflow(self):
        """
        Scenario: 5 neurons are passed but the hardware capacity maxes out at 4.
        Expectation: A HardwareConstraintError must be raised before loop finishes.
        """
        clustering = HierarchicalClusterOutput(
            cluster_assignment=np.array([0, 0, 1, 1, 2], dtype=np.int_),
            cluster_parent=np.array([-1], dtype=np.int_),
            num_clusters=3
        )

        mock_hw = MagicMock(spec=MosaicHardwareConfig)
        mock_hw.neurons_per_core = 2
        mock_hw.total_cores = 2

        with self.assertRaises(HardwareConstraintError) as context:
            self.adapter.adapt_clustering(clustering, mock_hw)

        self.assertIn("Hardware overflow", str(context.exception))

