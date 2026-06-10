import os
import pytest
import numpy as np

from netochi.input_generator.interfaces import MosaicHWMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.ilp_mapper import ILPMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapper
from netochi.mapping.three_step_mapping.hcd_pca_opt_three_step_mapper import HcdPcaOptThreeStepMapper
from netochi.mapping.three_step_mapping.hsbm_pca_opt_three_step_mapper import HsbmPcaOptThreeStepMapper
from netochi.visualization.visualize_clustering import plot_clustering_comparison
from netochi.visualization.visualize_mapping_output import plot_hardware_mapping
from tests.utils_graph_generation import create_fc_graph, create_two_fc_components, create_two_fc_with_directed_half


MAPPERS_TO_TEST = [
    SimAnnealingMapper(),
    # GreedyMapper(),
    # ILPMapper(),
    # QAPMapper(),
    # HcdPcaOptThreeStepMapper(),
    # HsbmPcaOptThreeStepMapper()
]


@pytest.fixture
def hw_config():
    """Provides a standard hardware configuration for the tests."""
    return MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=8,  # Must be an even number for Test 3
        router_levels=1,  # 2^1 = 2 total cores
        slice_factor=2
    )


@pytest.fixture
def output_dir():
    """Ensures the test results directory exists and returns its path."""
    out_dir = "test_results/"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# --- Test Cases ---

@pytest.mark.parametrize("mapper", MAPPERS_TO_TEST, ids=lambda m: m.__class__.__name__)
def test_fc_component_mapped_to_single_core(mapper, hw_config, output_dir):
    """
    A fully connected graph with size == neurons_per_core should be packed into a single core.
    """
    g = create_fc_graph(hw_config.neurons_per_core)
    mapping_input = MosaicHWMappingInput(graph=g, descriptions={}, hw_config=hw_config)

    state = mapper.run(mapping_input)
    unique_cores = np.unique(state.c)

    # Assertions
    mapper_name = mapper.__class__.__name__
    assert len(unique_cores) == 1, f"[{mapper_name}] Expected 1 core to be used, but found: {unique_cores}"

    # Visualization
    filename = os.path.join(output_dir, f"{mapper_name}_fc_component_mapped_to_single_core_MAPPING_STATE.pdf")
    plot_hardware_mapping(g, state, hw_config, filename=filename)


@pytest.mark.parametrize("mapper", MAPPERS_TO_TEST, ids=lambda m: m.__class__.__name__)
def test_two_fc_components_mapped_separately(mapper, hw_config, output_dir):
    """
    Two disjoint fully connected components should be cleanly separated into the two available cores.
    """
    N = hw_config.neurons_per_core
    g = create_two_fc_components(N)
    mapping_input = MosaicHWMappingInput(graph=g, descriptions={}, hw_config=hw_config)

    state = mapper.run(mapping_input)

    cores_A = np.unique(state.c[:N])
    cores_B = np.unique(state.c[N:2 * N])

    # Assertions
    mapper_name = mapper.__class__.__name__
    assert len(cores_A) == 1, f"[{mapper_name}] Cluster A is fragmented across multiple cores."
    assert len(cores_B) == 1, f"[{mapper_name}] Cluster B is fragmented across multiple cores."
    assert cores_A[0] != cores_B[0], f"[{mapper_name}] Both clusters were assigned to the same core."

    # Visualization
    filename = os.path.join(output_dir, f"{mapper_name}_two_fc_components_mapped_separately_MAPPING_STATE.pdf")
    plot_hardware_mapping(g, state, hw_config, filename=filename)


@pytest.mark.parametrize("mapper", MAPPERS_TO_TEST, ids=lambda m: m.__class__.__name__)
def test_slice_alignment_for_dependent_clusters(mapper, hw_config, output_dir):
    """
    Two FC components. Neuron 0 in core A listens to half of B.
    They must be separated, AND this half of B needs to be in the same slice.
    """
    N = hw_config.neurons_per_core
    g = create_two_fc_with_directed_half(N)
    mapping_input = MosaicHWMappingInput(graph=g, descriptions={}, hw_config=hw_config)

    state = mapper.run(mapping_input)

    cores_A = np.unique(state.c[:N])
    cores_B = np.unique(state.c[N:2 * N])
    mapper_name = mapper.__class__.__name__

    # 1. Check strict separation
    assert len(cores_A) == 1, f"[{mapper_name}] Cluster A is fragmented."
    assert len(cores_B) == 1, f"[{mapper_name}] Cluster B is fragmented."
    assert cores_A[0] != cores_B[0], f"[{mapper_name}] Clusters must be on different cores."

    # 2. Check slice consensus in A
    dist = hw_config.core_distance(cores_A[0], cores_B[0])
    assert dist == 1, f"[{mapper_name}] With router_levels=1, distance should inherently be 1."

    # 3. Check that the second half of B is neatly packed into a single slice
    half_n = N // 2
    slices_second_half_B = []
    for b in range(N + half_n, 2 * N):
        local_x = state.x[b]
        for sigma in range(hw_config.num_slices_at_distance(dist)):
            start_x, end_x = hw_config.get_slice_bounds(dist, sigma)
            if start_x <= local_x < end_x:
                slices_second_half_B.append(sigma)
                break

    unique_slices_second_half_B = np.unique(slices_second_half_B)
    assert len(unique_slices_second_half_B) == 1, (
        f"[{mapper_name}] The second half of Cluster B is split across multiple slices: "
        f"{unique_slices_second_half_B}. Expected it to be entirely aligned within one slice."
    )

    # --- Visualizations ---
    plot_hardware_mapping(
        g, state, hw_config,
        filename=os.path.join(output_dir, f"{mapper_name}_slice_alignment_for_dependent_clusters_MAPPING_STATE.pdf")
    )

    initial_assignment = np.zeros(2 * N, dtype=int)
    initial_assignment[N: 2 * N] = 1
    inferred_assignment = state.c

    plot_clustering_comparison(
        g=g,
        initial_assignment=initial_assignment,
        inferred_assignment=inferred_assignment,
        filename=os.path.join(output_dir, f"{mapper_name}_slice_alignment_for_dependent_clusters_CLUSTERING.pdf"),
        cluster_spread=1.5
    )