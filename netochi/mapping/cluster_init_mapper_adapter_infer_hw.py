
from typing import Any

from netochi.mapping.interfaces import BaseMapper, HWNetworkMappingState
from netochi.input_generator.interfaces import MappingInput

from netochi.mapping.three_step_mapping.interfaces import ClustererInferHw


class ClusterInitMapper(BaseMapper[HWNetworkMappingState[Any], MappingInput[Any]]):
    """
    Given a Clusterer and a mapper, HwClusterer and a Mapper, it computes the clustering and passes the clustering to the mapper as initialization
    """

    def __init__(self, clusterer: ClustererInferHw, mapper: BaseMapper[HWNetworkMappingState, MappingInput]):
        self.clusterer = clusterer
        self.mapper = mapper

    def run(self, mapping_input: MappingInput) -> HWNetworkMappingState[Any]:
        # --- 1. compute cluster initialization ---
        clustering = self.clusterer.cluster(mapping_input)
        mapping_input.core_assignment_initialization = clustering
        return self.mapper.run(mapping_input)

