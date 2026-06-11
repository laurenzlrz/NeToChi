
from typing import Any

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicHWMappingInput


from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw


class HwClusterInitMapper(BaseMapper[MosaicNetworkMappingState[Any], MosaicHWMappingInput[Any]]):
    """
    Given a HwClusterer and a Mapper, it computes the clustering and passes the clustering to the mapper as initialization
    """

    def __init__(self, clusterer: ClustererFixedHw, mapper: BaseMapper[MosaicNetworkMappingState, MosaicHWMappingInput]):
        self.clusterer = clusterer
        self.mapper = mapper

    def run(self, mapping_input: MosaicHWMappingInput) -> MosaicNetworkMappingState[Any]:
        # --- 1. compute cluster initialization ---
        clustering = self.clusterer.cluster(mapping_input)
        mapping_input.core_assignment_initialization = clustering.cluster_assignment
        return self.mapper.run(mapping_input)

