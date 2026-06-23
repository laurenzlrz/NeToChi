from pydantic import BaseModel
import icontract

from netochi.mapping.three_step_mapping.clustering.qap_hw_clusterer import QapHwClusterer
from netochi.mapping.three_step_mapping.local_address_assignment.pca_local_address_assigner import \
    PcaLocalAddressAssigner
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner
from netochi.mapping.three_step_mapping.three_step_hw_mapper import ThreeStepHwMapper


class QAPPcaOptMapperConfig(BaseModel):
    def create(self) -> "QAPPcaOptMapper":
        return QAPPcaOptMapper(config=self)


class QAPPcaOptMapper(ThreeStepHwMapper):
    """
    Heuristic mapper combining initial greedy clustering with random refinements.
    """

    @icontract.require(lambda config: isinstance(config, QAPPcaOptMapperConfig))
    def __init__(self, config: QAPPcaOptMapperConfig):
        self.config = config
        super().__init__(clusterer=QapHwClusterer(), address_assigner=PcaLocalAddressAssigner(),
                         slice_assigner=OptimalSliceAssigner())

