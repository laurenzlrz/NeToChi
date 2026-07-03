import numpy as np
import numpy.typing as npt

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import (
    ClusterAndHwOutput,
    ClustererFixedHw,
)


class GroundTruthOrRandomClusterer(ClustererFixedHw[MosaicMappingInput]):
    """
    Core clusterer that returns the ground truth core assignment if available.
    Otherwise, it assigns neurons to random clusters uniformly.
    Outputs a valid ClusterAndHwOutput configuration.
    """

    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        # Extract hardware config from the input data
        # (Adjust to input_data.hw if your HWMappingInput interface names it differently)
        hw_config = input_data.hw_config

        # 1. Try to use the ground truth core assignment if available
        if input_data.assignment is not None:
            cluster_assignment = np.array(
                input_data.assignment.neuron_core_pre_assignment, dtype=np.int_
            )
            # Determine total clusters based on the maximum cluster index assigned
            num_clusters = int(np.max(cluster_assignment) + 1) if cluster_assignment.size > 0 else 0

        # 2. Fallback: Random core assignment
        else:
            num_neurons = input_data.graph.num_vertices()
            num_clusters = 4

            cluster_assignment = np.random.randint(
                0, num_clusters, size=num_neurons, dtype=np.int_
            )

        # Return the expected structured format
        return ClusterAndHwOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=num_clusters,
            hw=hw_config
        )

    def get_name(self) -> str:
        return "Gt"