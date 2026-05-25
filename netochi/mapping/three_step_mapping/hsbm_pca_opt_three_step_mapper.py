
from netochi.mapping.three_step_mapping.clustering.hsbm_hw_clusterer import HsbmHwClusterer
from netochi.mapping.three_step_mapping.local_address_assignment.pca_local_address_assigner import PcaLocalAddressAssigner
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner
from netochi.mapping.three_step_mapping.three_step_hw_mapper import ThreeStepHwMapper


class HsbmPcaOptThreeStepMapper(ThreeStepHwMapper):
    """
    Heuristic mapper combining initial greedy clustering with random refinements.
    
    """

    def __init__(self):
        super().__init__(clusterer=HsbmHwClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner())

