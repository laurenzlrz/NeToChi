
from typing import Any

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicMappingInput


from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw


class HwClusterInitMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Given a HwClusterer and a Mapper, it computes the clustering and passes the clustering to the mapper as initialization
    """

    def __init__(self, clusterer: ClustererFixedHw, mapper: BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
        self.clusterer = clusterer
        self.mapper = mapper

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState[Any]:
        # --- 1. compute cluster initialization ---
        #TODO: IMPROVE ERROR VALIDATION
        clustering = self.clusterer.cluster(mapping_input)
        if (clustering.hw != mapping_input.hw_config):
            raise ValueError(f"Clusterer returned a different hardware configuration than the input. Clusterer hw: {clustering.hw}, input hw: {mapping_input.hw_config}")

        # TODO: IMPROVE INITIALIZATION.
        mapping_input = MosaicMappingInput(
            graph=mapping_input.graph,
            hw_config=clustering.hw,
            core_assignment_initialization=clustering)
        return self.mapper.run(mapping_input)

