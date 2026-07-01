import numpy as np

from netochi.input_generator.interfaces import MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory, MosaicNetworkConfig
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput
from netochi.mapping.three_step_mapping.local_address_assignment.lp_local_address_assigner import PipageRoundingAssigner
from netochi.objectives.utils import compute_e_valid

def test_lp_local_address_assigner():
    HW_SMALL = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=15,
        router_levels=2,
        slice_factor=2
    )

    PROB = 0.5

    FACTORY = MosaicNetworkConfig(hw_config=HW_SMALL, probability=PROB, seed=420, permute=True).create()
    INPUT = FACTORY.generate()
    assert INPUT.assignment is not None, "The generated input must have a valid assignment."

    CLUSTER_AND_HW_INPUT = ClusterAndHwOutput(cluster_assignment=INPUT.assignment.neuron_core_pre_assignment,
                                              num_clusters=np.max(INPUT.assignment.neuron_core_pre_assignment) + 1,
                                              hw=HW_SMALL)

    LP_LOCAL_ADDRESS_ASSIGNER = PipageRoundingAssigner()
    LOCAL_IDXS, SLICES = LP_LOCAL_ADDRESS_ASSIGNER.assign_addresses_and_slices(INPUT.graph, CLUSTER_AND_HW_INPUT)
    LP_ASSIGNMENT = MosaicAssignment(hw=HW_SMALL,
                                     neuron_core_pre_assignment=INPUT.assignment.neuron_core_pre_assignment,
                                     neuron_idx_pre_assignment=LOCAL_IDXS,
                                     neuron_slice_assignment=SLICES)
    LP_ASSIGNMENT.validate_graph(g=INPUT.graph)

    GROUND_TRUTH_ASSIGNMENT = INPUT.assignment
    GT_NUM_INCONSISTENCIES = INPUT.graph.num_edges() - compute_e_valid( # Do the Rest

    LP_ASSIGNMENT = #GET
    LP_NUM_INCONSISTENCIES

    #EVALUATE


