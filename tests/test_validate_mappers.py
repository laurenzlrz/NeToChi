import numpy as np
from graph_tool.collection import descriptions

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.ilp_mapper import ILPMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.visualization.visualize_clustering import plot_clustering_comparison
from netochi.visualization.visualize_mapping_output import plot_hardware_mapping
from tests.utils_graph_generation import create_fc_graph, create_two_fc_components, create_two_fc_with_directed_half


def check(condition, message, mapper):
    """
    Lightweight replacement for assertions.
    Prints a warning instead of stopping execution.
    """
    mapper_name = mapper.__class__.__name__ if mapper is not None else "UnknownMapper"
    if not condition:
        print(f"[FAILED][{mapper_name}] {message}")



def fc_component_mapped_to_single_core(mapper, hw_config, file_path):
    """
    Test Case 1:
    A fully connected graph with size == neurons_per_core should be packed into a single core.
    """
    g = create_fc_graph(hw_config.neurons_per_core)

    mapping_input = MosaicMappingInput(graph=g, descriptions = {}, hw_config=hw_config)
    state = mapper.run(mapping_input)

    # check all neurons are assigned to the exact same core
    unique_cores = np.unique(state.c)
    check(len(unique_cores) == 1, f"Expected 1 core to be used, but found: {unique_cores}", mapper)
    plot_hardware_mapping(g, state, hw_config, filename = file_path + "fc_component_mapped_to_single_core_MAPPING_STATE.pdf")


def two_fc_components_mapped_separately(mapper, hw_config, file_path):
    """
    Test Case 2:
    Two disjoint fully connected components should be cleanly separated into the two available cores.
    """
    N = hw_config.neurons_per_core
    g = create_two_fc_components(N)

    mapping_input = MosaicMappingInput(graph=g, descriptions = {}, hw_config=hw_config)
    state = mapper.run(mapping_input)

    cores_A = np.unique(state.c[:N])
    cores_B = np.unique(state.c[N:2 * N])

    check(len(cores_A) == 1, "Cluster A is fragmented across multiple cores.", mapper)
    check(len(cores_B) == 1, "Cluster B is fragmented across multiple cores.", mapper)
    check(cores_A[0] != cores_B[0], "Both clusters were assigned to the same core.", mapper)
    plot_hardware_mapping(g, state, hw_config, filename = file_path  + "two_fc_components_mapped_separately_MAPPING_STATE.pdf")



def slice_alignment_for_dependent_clusters(mapper, hw_config, file_path):
    """
    Test Case 3:
    Two FC components. Neuron 0 in core A listens to half of B.
    They must be separated, AND this half of B needs to be in same slice.
    """
    N = hw_config.neurons_per_core
    g = create_two_fc_with_directed_half(N)

    mapping_input = MosaicMappingInput(graph=g, descriptions = {}, hw_config=hw_config)
    state = mapper.run(mapping_input)

    cores_A = np.unique(state.c[:N])
    cores_B = np.unique(state.c[N:2 * N])

    plot_hardware_mapping(g, state, hw_config, filename = file_path + "slice_alignment_for_dependent_clusters_MAPPING_STATE.pdf")

    # --- ADDED: Visualizing the structural clustering comparison ---
    # Construct the ground-truth assignment: 0 for Cluster A elements, 1 for Cluster B elements
    initial_assignment = np.zeros(2 * N, dtype=int)
    initial_assignment[N: 2 * N] = 1

    # Inferred assignment is the actual core placement chosen by the ILP solver
    inferred_assignment = state.c

    plot_clustering_comparison(
        g=g,
        initial_assignment=initial_assignment,
        inferred_assignment=inferred_assignment,
        filename=file_path + "slice_alignment_for_dependent_clusters_CLUSTERING.pdf",
        cluster_spread=1.5
    )
    # ---------------------------------------------------------------
    # 1. Check strict separation
    check(len(cores_A) == 1, "Cluster A is fragmented.", mapper)
    check(len(cores_B) == 1, "Cluster B is fragmented.", mapper)
    check( cores_A[0] != cores_B[0], "Clusters must be on different cores.", mapper)

    # 2. Check slice consensus in A
    # Get the routing distance between Core A and Core B
    dist = hw_config.core_distance(cores_A[0], cores_B[0])
    check( dist == 1, "With router_levels=1, distance should inherently be 1.", mapper)

    # 3. check that the second half of B is neatly packed into a single slice
    half_n = N // 2
    slices_second_half_B = []
    for b in range(N + half_n, 2 * N):
        local_x = state.x[b]
        # Resolve which hardware slice window this local index falls into
        for sigma in range(hw_config.num_slices_at_distance(dist)):
            start_x, end_x = hw_config.get_slice_bounds(dist, sigma)
            if start_x <= local_x < end_x:
                slices_second_half_B.append(sigma)
                break

    unique_slices_second_half_B = np.unique(slices_second_half_B)
    check( len(unique_slices_second_half_B) == 1, (
        f"The second half of Cluster B is split across multiple slices: {unique_slices_second_half_B}. "
        f"Expected it to be entirely aligned within one slice."
    ), mapper)


def test_validate_mappers_with_given_hw():
    mappers = {ILPMapper()}
    tests = {fc_component_mapped_to_single_core, two_fc_components_mapped_separately, slice_alignment_for_dependent_clusters}
    hw_config = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=8,  # Must be an even number for Test 3
        router_levels=1,  # 2^1 = 2 total cores
        slice_factor=2
    )
    file_path = "test_results/test_slice_alignment_for_dependent_clusters.pdf"
    print("\n")
    for mapper in mappers:
        for test_function in tests:
            test_function(mapper, hw_config, file_path)



