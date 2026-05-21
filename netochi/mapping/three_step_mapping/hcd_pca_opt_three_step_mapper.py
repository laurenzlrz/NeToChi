from netochi.mapping.three_step_mapping.clustering.hcd_hw_clusterer import HcdHwClusterer
from netochi.mapping.three_step_mapping.local_address_assignment.pca_local_address_assigner import \
    PcaLocalAddressAssigner
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner
from netochi.mapping.three_step_mapping.three_step_mapper import ThreeStepMapper




class HcdPcaOptThreeStepMapper(ThreeStepMapper):

    def __init__(self):
        super().__init__(clusterer=HcdHwClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner())