from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, ClusterAndHwOutput, \
    HierarchicalClusterer, ClusteringAdapter, ClustererInferHw


class HwClustererAdapter(ClustererInferHw):
    """
    This adapter runs a HierarchicalClusterer and then transforms the HierarchicalClusterOutput into ClusterAndHwOutput, i.e. it infers the hardware and adapts the clustering, so that it fits the hardware

    input: MappingInput

    1. runs hierarchical clustering
    2. infers hardware
    3. adapts clustering, so that it fits the hardware

    output: ClusterAndHWOutput
    """

    def __init__(self, clusterer: HierarchicalClusterer, adapter: ClusteringAdapter) -> None:
        self._clusterer = clusterer
        self._adapter = adapter


    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        clustering: HierarchicalClusterOutput = self._clusterer.cluster(input_data=input_data)
        adapted_clustering: ClusterAndHwOutput = self._adapter.adapt_clustering(clustering=clustering)
        return adapted_clustering

