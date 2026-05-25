from netochi.input_generator.interfaces import HWMappingInput
from netochi.mapping.three_step_mapping.clustering.cluster_adapter.fill_given_hw_adapter import FillGivenHwAdapter
from netochi.mapping.three_step_mapping.clustering.clusterer.hsbm_clusterer import HsbmClusterer
from netochi.mapping.three_step_mapping.clustering.hw_clusterer_adapter_given_hw import GivenHwClustererAdapter
from netochi.mapping.three_step_mapping.interfaces import HwClusterer, ClusterAndHwOutput


class HsbmHwClusterer(HwClusterer):
    """
    1. infers clustering using the Hsbm
    2. fits clustering to given hardware using the FillGivenHwAdapter
    """

    def __init__(self):
        self._clusterer = GivenHwClustererAdapter(clusterer=HsbmClusterer(), adapter=FillGivenHwAdapter())

    def cluster(self, input_data: HWMappingInput) -> ClusterAndHwOutput:
        return self._clusterer.cluster(input_data)

