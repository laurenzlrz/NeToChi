
import numpy as np
from typing import Any

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment


from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw


class HwClusterInitMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Given a HwClusterer and a Mapper, it computes the clustering and passes the clustering to the mapper as initialization
    """

    def __init__(self, clusterer: ClustererFixedHw, mapper: BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
        self.clusterer = clusterer
        self.mapper = mapper

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        # --- 1. compute cluster initialization ---
        #TODO: IMPROVE ERROR VALIDATION
        clustering = self.clusterer.cluster(mapping_input)
        if (clustering.hw != mapping_input.hw_config):
            raise ValueError(f"Clusterer returned a different hardware configuration than the input. Clusterer hw: {clustering.hw}, input hw: {mapping_input.hw_config}")

        # TODO: IMPROVE INITIALIZATION.
        num_neurons = mapping_input.graph.num_vertices()
        assignment = MosaicAssignment(
            hw=clustering.hw,
            neuron_core_pre_assignment=clustering.cluster_assignment.astype(np.int64),
            neuron_idx_pre_assignment=np.zeros(num_neurons, dtype=np.int64),
            neuron_slice_assignment=np.zeros((num_neurons, clustering.hw.router_levels + 1), dtype=np.int64)
        )
        mapping_input = MosaicMappingInput(
            graph=mapping_input.graph,
            hw_config=clustering.hw,
            descriptions=mapping_input.descriptions,
            assignment=assignment)
        return self.mapper.run(mapping_input)


