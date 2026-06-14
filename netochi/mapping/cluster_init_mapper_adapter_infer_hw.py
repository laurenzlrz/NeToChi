
import numpy as np
from typing import Any

from netochi.mapping.interfaces import BaseMapper, MosaicHWMappingState
from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, MosaicAssignment

from netochi.mapping.three_step_mapping.interfaces import ClustererInferHw


class ClusterInitMapper(BaseMapper[MosaicHWMappingState, MappingInput]):
    """
    Given a Clusterer and a mapper, HwClusterer and a Mapper, it computes the clustering and passes the clustering to the mapper as initialization
    """

    def __init__(self, clusterer: ClustererInferHw, mapper: BaseMapper[MosaicHWMappingState, MappingInput]):
        self.clusterer = clusterer
        self.mapper = mapper

    def run(self, mapping_input: MappingInput) -> MosaicHWMappingState:
        # --- 1. compute cluster initialization ---
        clustering = self.clusterer.cluster(mapping_input)
        num_neurons = mapping_input.graph.num_vertices()
        assignment = MosaicAssignment(
            hw=clustering.hw,
            neuron_core_pre_assignment=clustering.cluster_assignment.astype(np.int64),
            neuron_idx_pre_assignment=np.zeros(num_neurons, dtype=np.int64),
            neuron_slice_assignment=np.zeros((num_neurons, clustering.hw.router_levels + 1), dtype=np.int64)
        )
        mapping_input_adapted = MosaicMappingInput(
            graph=mapping_input.graph,
            hw_config=clustering.hw,
            descriptions=mapping_input.descriptions,
            assignment=assignment)
        return self.mapper.run(mapping_input_adapted)

