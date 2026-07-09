import icontract
from pydantic import BaseModel, Field

from netochi.mapping.cluster_init_mapper_adapter_fix_hw import HwClusterInitMapper
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapperConfig
from netochi.mapping.three_step_mapping.clustering.clusterer.bottom_up_clusterer import BottomUpClusterer
from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import KaMinParHierarchicalClusterer, \
    kaMinParClusterFunction


class SimulatedAnnealingMapperClusterInitConfig(BaseModel):
    verbose: bool = Field(default=False)

    def create(self) -> "SimulatedAnnealingMapperClusterInit":
        return SimulatedAnnealingMapperClusterInit(config=self)


class SimulatedAnnealingMapperClusterInit(HwClusterInitMapper):
    """
    Given a HwClusterer and a Mapper, it computes a clustering that fits on the hardware and passes the clustering to the mapper as initialization
    """

    @icontract.require(lambda config: isinstance(config, SimulatedAnnealingMapperClusterInitConfig))
    def __init__(self, config: SimulatedAnnealingMapperClusterInitConfig):
        clusterer = BottomUpClusterer(clusterer=kaMinParClusterFunction)
        mapper = SimAnnealingMapperConfig().create()
        super().__init__(clusterer, mapper)


