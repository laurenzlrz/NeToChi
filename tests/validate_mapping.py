import pytest
import numpy as np
import graph_tool.all as gt

from netochi.input_generator.interfaces import HWMappingInput, MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.ilp_mapper import ILPMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.visualization.visualize_clustering import plot_clustering_comparison
from netochi.visualization.visualize_mapping_output import plot_hardware_mapping


# Import your classes here (assuming they are in a module named 'mosaic_mapping')
# from mosaic_mapping import MosaicHardwareConfig, HWMappingInput, MosaicNetworkMappingState

# ==========================================
# Graph Generation Helpers
# ==========================================

def create_fc_graph(n: int) -> gt.Graph:
    """Creates a single fully connected component of size n."""
    g = gt.Graph(directed=True)
    g.add_vertex(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                g.add_edge(i, j)
    return g


def create_two_fc_components(n: int) -> gt.Graph:
    """
    n: number neurons per component
    Creates two disconnected fully connected components of size n each.
    """
    g = gt.Graph(directed=True)
    g.add_vertex(2 * n)

    # Component A (Indices 0 to n-1)
    for i in range(n):
        for j in range(n):
            if i != j: g.add_edge(i, j)

    # Component B (Indices n to 2n-1)
    for i in range(n, 2 * n):
        for j in range(n, 2 * n):
            if i != j: g.add_edge(i, j)

    return g


def create_two_fc_with_directed_half(n: int) -> gt.Graph:
    """
    n: number neurons per component
    Creates two FC components (A and B).
    Exactly half of B connects to all neurons in A. No other inter-cluster edges.
    """
    assert n % 2 == 0, "n must be even to cleanly split half of cluster B."
    g = create_two_fc_components(n)

    # Half of B (Indices n to n + n//2 - 1) connects to all of A (Indices 0 to n-1)
    half_n = n // 2
    for b in range(n, n + half_n):
        g.add_edge(b, 0) # node ß receives input from half of b -> should have slice idx = 0

    return g


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def hw_config() -> MosaicHardwareConfig:
    """Provides a hardware config with exactly 2 cores."""
    return MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=8,  # Must be an even number for Test 3
        router_levels=1,  # 2^1 = 2 total cores
        slice_factor=2
    )


@pytest.fixture
def mapper():
    """
    Provide your instantiated mapper here.
    Example: return MyCustomMapper()
    """
    return QAPMapper()


# ==========================================
# Tests
# ==========================================
class TestMapper:
    def test_fc_component_mapped_to_single_core(self, mapper, hw_config):
        """
        Test Case 1:
        A fully connected graph with size == neurons_per_core should be packed into a single core.
        """
        N = hw_config.neurons_per_core
        g = create_fc_graph(N)

        mapping_input = MosaicMappingInput(graph=g, descriptions={}, hw_config=hw_config)
        state = mapper.run(mapping_input)

        # Assert all neurons are assigned to the exact same core
        unique_cores = np.unique(state.c)
        assert len(unique_cores) == 1, f"Expected 1 core to be used, but found: {unique_cores}"
        plot_hardware_mapping(g, state, hw_config, filename = "../netochi/results/test_fc_component_mapped_to_single_core.pdf")


    def test_two_fc_components_mapped_separately(self, mapper, hw_config):
        """
        Test Case 2:
        Two disjoint fully connected components should be cleanly separated into the two available cores.
        """
        N = hw_config.neurons_per_core
        g = create_two_fc_components(N)

        mapping_input = MosaicMappingInput(graph=g, descriptions={}, hw_config=hw_config)
        state = mapper.run(mapping_input)

        cores_A = np.unique(state.c[:N])
        cores_B = np.unique(state.c[N:2 * N])

        # Assert clean separation
        assert len(cores_A) == 1, "Cluster A is fragmented across multiple cores."
        assert len(cores_B) == 1, "Cluster B is fragmented across multiple cores."
        assert cores_A[0] != cores_B[0], "Both clusters were assigned to the same core."
        plot_hardware_mapping(g, state, hw_config, filename = "../netochi/results/test_two_fc_components_mapped_separately.pdf")



    def test_slice_alignment_for_dependent_clusters(self, mapper, hw_config):
        """
        Test Case 3:
        Two FC components. A listens to half of B.
        They must be separated, AND all neurons in A must select the same listening slice from B.
        """
        N = hw_config.neurons_per_core
        g = create_two_fc_with_directed_half(N)

        mapping_input = MosaicMappingInput(graph=g, descriptions={}, hw_config=hw_config)
        state = mapper.run(mapping_input)

        cores_A = np.unique(state.c[:N])
        cores_B = np.unique(state.c[N:2 * N])

        plot_hardware_mapping(g, state, hw_config, filename = "../netochi/results/test_slice_alignment_for_dependent_clusters.pdf")

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
            filename="../netochi/results/test_slice_alignment_clustering_comparison.pdf",
            cluster_spread=1.5
        )
        # ---------------------------------------------------------------
        # 1. Check strict separation
        assert len(cores_A) == 1, "Cluster A is fragmented."
        assert len(cores_B) == 1, "Cluster B is fragmented."
        assert cores_A[0] != cores_B[0], "Clusters must be on different cores."

        # 2. Check slice consensus in A
        # Get the routing distance between Core A and Core B
        dist = hw_config.core_distance(cores_A[0], cores_B[0])
        assert dist == 1, "With router_levels=1, distance should inherently be 1."

        # 3. Assert that the second half of B is neatly packed into a single slice
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
        assert len(unique_slices_second_half_B) == 1, (
            f"The second half of Cluster B is split across multiple slices: {unique_slices_second_half_B}. "
            f"Expected it to be entirely aligned within one slice."
        )

        # 4. Assert that Neuron 0 explicitly listens to slice 1
        slice_neuron_0 = state.s[0, dist]
        assert slice_neuron_0 == 0, (
            f"Neuron 0 is listening to slice {slice_neuron_0} at distance {dist}, "
            f"but it was expected to listen to slice 1."
        )
