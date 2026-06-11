
from pydantic import PrivateAttr
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, ClusterAndHwOutput, \
    HierarchicalClusterer, ClusteringAdapterFixedHw, ClustererFixedHw


class GivenHwClustererAdapter(ClustererFixedHw):
    """
    Given a hardware, this adapter runs a HierarchicalClusterer and then transforms the HierarchicalClusterOutput into ClusterAndHwOutput, i.e. it adapts the clustering, so that it fits the hardware

    input: HwMappingInput

    1. runs hierarchical clustering
    2. infers hardware
    3. adapts clustering, so that it fits the hardware

    output: ClusterAndHWOutput
    """
    _clusterer: HierarchicalClusterer = PrivateAttr()
    _adapter: ClusteringAdapterFixedHw = PrivateAttr()

    def __init__(self, clusterer: HierarchicalClusterer, adapter: ClusteringAdapterFixedHw):
        super().__init__()
        self._clusterer = clusterer
        self._adapter = adapter

    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        clustering: HierarchicalClusterOutput = self._clusterer.cluster(input_data=input_data)
        adapted_clustering: ClusterAndHwOutput = self._adapter.adapt_clustering(clustering=clustering, hw_config=input_data.hw_config)
        return adapted_clustering

