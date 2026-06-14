from pydantic import PrivateAttr
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.three_step_mapping.clustering.cluster_adapter.padding_adapter import PaddingClusteringAdapter
from netochi.mapping.three_step_mapping.clustering.clusterer.hcd_clusterer import HcdClusterer
from netochi.mapping.three_step_mapping.clustering.hw_clusterer_adapter import HwClustererAdapter
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput, ClustererInferHw


class HcdHwClusterer(ClustererInferHw):
    """
    1. infers clustering using the Hierarchical Community Detection
    2. infers hardware and fits clustering to it using the padding adapter
    """
    _clusterer: HwClustererAdapter = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._clusterer = HwClustererAdapter(clusterer=HcdClusterer(), adapter=PaddingClusteringAdapter())

    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        return self._clusterer.cluster(input_data)

