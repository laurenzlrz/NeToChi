import numpy as np

from netochi.input_generator.interfaces import MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory, MosaicNetworkConfig
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput
from netochi.mapping.three_step_mapping.local_address_assignment.lp_local_address_assigner import PipageRoundingAssigner
from netochi.mapping.interfaces import MosaicNetworkMappingState
from netochi.objectives.utils import compute_e_valid


def test_lp_local_address_assigner():
    HW_SMALL = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=15,
        router_levels=2,
        slice_factor=2
    )

    seeds = [42, 100, 420, 660, 235, 2356, 13527, 1366]
    probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "="*80)
    print(" LP Local Address Assigner Benchmark Summary ".center(80, "="))
    print("="*80)
    print(f"{'Seed':^6} | {'Prob':^6} | {'Total Edges':^13} | {'GT Incons':^11} | {'LP Incons':^11} | {'Diff Rate':^11} | {'Mark':^12}")
    print("-"*80)

    any_large_diff = False

    for seed in seeds:
        for prob in probabilities:
            FACTORY = MosaicNetworkConfig(hw_config=HW_SMALL, probability=prob, seed=seed, permute=True).create()
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

            # Prepare data for compute_e_valid
            graph_data = {
                'N': INPUT.graph.num_vertices(),
                'm': INPUT.graph.num_edges(),
                'in_edges': [[int(src) for src in v.in_neighbors()] for v in INPUT.graph.vertices()]
            }

            # Evaluate ground truth assignment
            GROUND_TRUTH_ASSIGNMENT = INPUT.assignment
            gt_state = MosaicNetworkMappingState(assignment=GROUND_TRUTH_ASSIGNMENT, mapping_input=INPUT)
            GT_NUM_INCONSISTENCIES = INPUT.graph.num_edges() - compute_e_valid(gt_state, graph_data)

            # Evaluate LP assignment
            lp_state = MosaicNetworkMappingState(assignment=LP_ASSIGNMENT, mapping_input=INPUT)
            LP_NUM_INCONSISTENCIES = INPUT.graph.num_edges() - compute_e_valid(lp_state, graph_data)

            total_edges = INPUT.graph.num_edges()
            diff_inconsistencies = LP_NUM_INCONSISTENCIES - GT_NUM_INCONSISTENCIES
            diff_rate = diff_inconsistencies / total_edges if total_edges > 0 else 0.0

            mark = ""
            if diff_rate > 0.5:
                mark = "⚠️ DIFF > 0.5"
                any_large_diff = True

            print(f"{seed:^6} | {prob:^6.1f} | {total_edges:^13} | {GT_NUM_INCONSISTENCIES:^11} | {LP_NUM_INCONSISTENCIES:^11} | {diff_rate:^11.4f} | {mark:^12}")

            assert LP_NUM_INCONSISTENCIES >= 0, "Inconsistencies count must be non-negative"

    print("="*80)
    if any_large_diff:
        print("Note: In some cases, the LP Assigner had a difference rate greater than 0.5.")
    else:
        print("Success: LP Assigner difference rate is below 0.5 in all tested configurations.")




