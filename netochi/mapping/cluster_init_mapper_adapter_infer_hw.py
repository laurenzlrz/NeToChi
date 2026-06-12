
from typing import Any

from netochi.mapping.interfaces import BaseMapper, MosaicHWMappingState
from netochi.input_generator.interfaces import MappingInput

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
        mapping_input = MosaicMappingInput(
            graph=mapping_input.graph,
            hw_config=clustering.hw,
            core_assignment_initialization=clustering)
        return self.mapper.run(mapping_input)

