from collections import defaultdict

import numpy as np

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.definitions.exceptions import HardwareConstraintError
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, ClusterAndHwOutput, HwClusteringAdapter


class FillGivenHwAdapter(HwClusteringAdapter):
    """
    this adapter throws away the hierarchy and only takes the lowest-level cluster as leaves.
    It sorts the neurons by cluster id and fills the cores one by one until all neurons are placed.
    Clusters are not guaranteed to remain intact!!
    """

    def adapt_clustering(self, clustering: HierarchicalClusterOutput, hw_config: MosaicHardwareConfig) -> ClusterAndHwOutput:
        num_neurons = len(clustering.cluster_assignment)
        core_assignment = np.zeros(num_neurons, dtype=np.int_)

        nodes_per_cluster = defaultdict(list)
        for neuron_id, cluster_id in enumerate(clustering.cluster_assignment):
            nodes_per_cluster[cluster_id].append(neuron_id)

        current_core = 0
        current_count = 0
        for cluster in sorted(
                nodes_per_cluster.keys()):  # go through clusters in ascending order: adds nodes of current core to current node until core is full
            for node in nodes_per_cluster[cluster]:
                if current_count >= hw_config.neurons_per_core:  # once core is full: move to next core
                    current_core += 1
                    current_count = 0
                if current_core >= hw_config.total_cores:
                    raise HardwareConstraintError(
                        f"Hardware overflow: Number of neurons larger than number of available slots on chip."
                    )
                core_assignment[node] = current_core
                current_count += 1


        return ClusterAndHwOutput(
            cluster_assignment=core_assignment,
            num_clusters=hw_config.total_cores,
            hw=hw_config
        )


