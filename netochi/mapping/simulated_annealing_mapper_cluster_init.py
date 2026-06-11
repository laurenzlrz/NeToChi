from netochi.mapping.cluster_init_mapper_adapter_fix_hw import HwClusterInitMapper
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapper
from netochi.mapping.three_step_mapping.clustering.hsbm_hw_clusterer import HsbmHwClusterer


class SimulatedAnnealingMapperClusterInit(HwClusterInitMapper):
    """
    Given a HwClusterer and a Mapper, it computes a clustering that fits on the hardware and passes the clustering to the mapper as initialization
    """

    def __init__(self):
        clusterer = HsbmHwClusterer()
        mapper = SimAnnealingMapper()
        super().__init__(clusterer, mapper)


