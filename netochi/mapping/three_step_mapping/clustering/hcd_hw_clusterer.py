from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput
from netochi.mapping.three_step_mapping.clustering.cluster_adapter.fill_given_hw_adapter import FillGivenHwAdapter
from netochi.mapping.three_step_mapping.clustering.cluster_adapter.padding_adapter import PaddingClusteringAdapter
from netochi.mapping.three_step_mapping.clustering.clusterer.hcd_clusterer import HcdClusterer
from netochi.mapping.three_step_mapping.clustering.hw_clusterer_adapter import HwClustererAdapter
from netochi.mapping.three_step_mapping.clustering.hw_clusterer_adapter_given_hw import GivenHwClustererAdapter
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput, ClustererInferHw, ClustererFixedHw


class HcdInferHwClusterer(ClustererInferHw):
    """
    1. infers clustering using the Hierarchical Community Detection
    2. infers hardware and fits clustering to it using the padding adapter
    """

    def __init__(self) -> None:
        self._clusterer = HwClustererAdapter(clusterer=HcdClusterer(), adapter=PaddingClusteringAdapter())


    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        return self._clusterer.cluster(input_data)

    def get_name(self) -> str:
        return "HCD"


class HcdHwClusterer(ClustererFixedHw):
    """
    1. infers clustering using the Hierarchical Community Detection
    2. infers hardware and fits clustering to it using the padding adapter
    """

    def __init__(self) -> None:
        self._clusterer = GivenHwClustererAdapter(clusterer=HcdClusterer(), adapter=FillGivenHwAdapter())


    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        return self._clusterer.cluster(input_data)

    def get_name(self) -> str:
        return "HCD"



