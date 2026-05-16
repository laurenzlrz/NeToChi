
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig

import numpy as np
import graph_tool.all as gt
from dataclasses import dataclass

from netochi.visualization.visualize_adjacency_matrix import plot_sorted_adjacency
from netochi.visualization.visualize_clustering import plot_clustering_comparison
from netochi.visualization.visualize_mapping_output import plot_hardware_mapping
from netochi.visualization.visualize_routing_hierarchy import plot_routing_hierarchy


@dataclass
class MockMappingState:
    neuron_core_idxs_assignment: np.ndarray
    neuron_local_idxs_assignment: np.ndarray
    neuron_slice_assignments: np.ndarray


def test_hardware_mapping_visualization():
    # A. Define Hardware (4 cores, 8 neurons per core)
    config = MosaicHardwareConfig(
        nodes_per_router=2,
        router_levels=2,
        neurons_per_core=8,
        slice_factor=2
    )

    # B. Generate a test graph
    g = gt.Graph(directed=True)
    g.add_vertex(config.total_neurons)

    # 1. Create fully connected cores
    for c in range(config.total_cores):
        start_idx = c * config.neurons_per_core
        end_idx = start_idx + config.neurons_per_core
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j:
                    g.add_edge(i, j)

    # 2. Add exactly 8 edges from the same core to a single target neuron
    # We will use Core 0 as the source, and the first neuron of Core 1 as the target
    source_core = 0
    target_neuron = config.neurons_per_core  # Index of the first neuron in Core 1

    # Safely take up to 8 neurons from the source core
    num_inter_edges = min(8, config.neurons_per_core)

    for local_idx in range(num_inter_edges):
        source_neuron = (source_core * config.neurons_per_core) + local_idx
        g.add_edge(source_neuron, target_neuron)

    # 3. Add another 8 edges from a core at distance 2 to the SAME target neuron
    # Find a core ID that sits at a routing distance of 2 from Core 1
    source_core_dist2 = None
    target_core = 1

    for c in range(config.total_cores):
        if config.core_distance(c, target_core) == 2:
            source_core_dist2 = c
            break

    if source_core_dist2 is None:
        raise ValueError(
            "Could not find a core at distance 2. Ensure your hardware layout supports distance 2 routing.")

    # Add the 8 edges from the distance 2 core to the same target neuron
    for local_idx in range(num_inter_edges):
        source_neuron = (source_core_dist2 * config.neurons_per_core) + local_idx
        g.add_edge(source_neuron, target_neuron)

    # C. Create a Mock Mapping State
    cores = np.zeros(config.total_neurons, dtype=int)
    locals_idx = np.zeros(config.total_neurons, dtype=int)

    # Perfectly distribute neurons across all cores and local indices
    for i in range(config.total_neurons):
        cores[i] = i // config.neurons_per_core
        locals_idx[i] = i % config.neurons_per_core

    # Simulate random slice assignments for distances 0, 1, 2
    # Shape: [num_neurons, max_distance + 1]
    slices = np.zeros((config.total_neurons, config.max_distance + 1), dtype=int)
    for d in range(1, config.max_distance + 1):
        num_slices_available = config.num_slices_at_distance(d)
        # Randomly assign each neuron a slice index it "listens" to at this distance
        slices[:, d] = np.random.randint(0, num_slices_available, size=config.total_neurons)

    # set slice for target neuron
    slices[target_neuron, 1] = 0
    slices[target_neuron, 2] = 0

    state = MockMappingState(
        neuron_core_idxs_assignment=cores,
        neuron_local_idxs_assignment=locals_idx,
        neuron_slice_assignments=slices
    )

    # D. Generate the plot
    plot_hardware_mapping(g, state, config, filename="test_results/test_example_hw_mapping.pdf")


def test_hardware_mapping_visualization_more_router_levels():
    # A. Define Hardware (4 cores, 8 neurons per core)
    config = MosaicHardwareConfig(
        nodes_per_router=2,
        router_levels=3,
        neurons_per_core=8,
        slice_factor=2
    )

    # B. Generate a test graph
    g = gt.Graph(directed=True)
    g.add_vertex(config.total_neurons)

    # 1. Create fully connected cores
    for c in range(config.total_cores):
        start_idx = c * config.neurons_per_core
        end_idx = start_idx + config.neurons_per_core
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j:
                    g.add_edge(i, j)

    # 2. Add exactly 8 edges from the same core to a single target neuron
    # We will use Core 0 as the source, and the first neuron of Core 1 as the target
    source_core = 0
    target_neuron = config.neurons_per_core  # Index of the first neuron in Core 1

    # Safely take up to 8 neurons from the source core
    num_inter_edges = min(8, config.neurons_per_core)

    for local_idx in range(num_inter_edges):
        source_neuron = (source_core * config.neurons_per_core) + local_idx
        g.add_edge(source_neuron, target_neuron)

    # 3. Add another 8 edges from a core at distance 2 to the SAME target neuron
    # Find a core ID that sits at a routing distance of 2 from Core 1
    source_core_dist2 = None
    target_core = 1

    for c in range(config.total_cores):
        if config.core_distance(c, target_core) == 2:
            source_core_dist2 = c
            break

    if source_core_dist2 is None:
        raise ValueError(
            "Could not find a core at distance 2. Ensure your hardware layout supports distance 2 routing.")

    # Add the 8 edges from the distance 2 core to the same target neuron
    for local_idx in range(num_inter_edges):
        source_neuron = (source_core_dist2 * config.neurons_per_core) + local_idx
        g.add_edge(source_neuron, target_neuron)

    # C. Create a Mock Mapping State
    cores = np.zeros(config.total_neurons, dtype=int)
    locals_idx = np.zeros(config.total_neurons, dtype=int)

    # Perfectly distribute neurons across all cores and local indices
    for i in range(config.total_neurons):
        cores[i] = i // config.neurons_per_core
        locals_idx[i] = i % config.neurons_per_core

    # Simulate random slice assignments for distances 0, 1, 2
    # Shape: [num_neurons, max_distance + 1]
    slices = np.zeros((config.total_neurons, config.max_distance + 1), dtype=int)
    for d in range(1, config.max_distance + 1):
        num_slices_available = config.num_slices_at_distance(d)
        # Randomly assign each neuron a slice index it "listens" to at this distance
        slices[:, d] = np.random.randint(0, num_slices_available, size=config.total_neurons)

    # set slice for target neuron
    slices[target_neuron, 1] = 0
    slices[target_neuron, 2] = 0

    state = MockMappingState(
        neuron_core_idxs_assignment=cores,
        neuron_local_idxs_assignment=locals_idx,
        neuron_slice_assignments=slices
    )

    # D. Generate the plot
    plot_hardware_mapping(g, state, config, filename="test_results/test_example_hw_mapping.pdf")




def test_adjacency_matrix_visualization_only_between_cores_0_and_1():
    # A. Setup hardware parameters for the mock
    num_cores = 3
    neurons_per_core = 15
    total_neurons = num_cores * neurons_per_core

    # B. Create a graph with clear cluster structures (Stochastic Block Model approach)
    # We want intra-core connections to be dense (70%), inter-core to be sparse (5%)
    g = gt.Graph(directed=True)
    g.add_vertex(total_neurons)

    for i in range(total_neurons):
        for j in range(total_neurons):
            if i != j:
                core_i = i // neurons_per_core
                core_j = j // neurons_per_core

                if core_i == 0 and core_j == 1:
                    prob = 0.70 if core_i == core_j else 0.05
                    if np.random.rand() < prob:
                        g.add_edge(i, j)

    # C. Create assignments simulating a perfect mapping algorithm
    # To prove the sorting works, we will scramble the neurons before passing them in

    core_assignments = np.zeros(total_neurons, dtype=int)
    local_assignments = np.zeros(total_neurons, dtype=int)

    for neuron_id in range(total_neurons):
        core_assignments[neuron_id] = neuron_id // neurons_per_core
        local_assignments[neuron_id] = neuron_id % neurons_per_core

    mock_state = MockMappingState(
        neuron_core_idxs_assignment=core_assignments,
        neuron_local_idxs_assignment=local_assignments,
        neuron_slice_assignments=None
    )

    # D. Run the visualization!
    # Because our mock mapping matches the dense cluster generation,
    # you should see 3 dense black squares separated by red lines.
    plot_sorted_adjacency(g, mock_state, filename="test_results/adjacency_matrix.pdf")


def test_adjacency_matrix_visualization_only_between_neuron_0_and_neuron_15():
    # A. Setup hardware parameters for the mock
    num_cores = 3
    neurons_per_core = 15
    total_neurons = num_cores * neurons_per_core

    # B. Create a graph with clear cluster structures (Stochastic Block Model approach)
    # We want intra-core connections to be dense (70%), inter-core to be sparse (5%)
    g = gt.Graph(directed=True)
    g.add_vertex(total_neurons)

    g.add_edge(0,15)

    # C. Create assignments simulating a perfect mapping algorithm
    # To prove the sorting works, we will scramble the neurons before passing them in

    core_assignments = np.zeros(total_neurons, dtype=int)
    local_assignments = np.zeros(total_neurons, dtype=int)

    for neuron_id in range(total_neurons):
        core_assignments[neuron_id] = neuron_id // neurons_per_core
        local_assignments[neuron_id] = neuron_id % neurons_per_core

    mock_state = MockMappingState(
        neuron_core_idxs_assignment=core_assignments,
        neuron_local_idxs_assignment=local_assignments,
        neuron_slice_assignments=None
    )

    # D. Run the visualization!
    # Because our mock mapping matches the dense cluster generation,
    # you should see 3 dense black squares separated by red lines.
    plot_sorted_adjacency(g, mock_state, filename="test_results/adjacency_matrix.pdf")



def test_routing_hierarchy_visualization():
    hw_config = MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=256, router_levels=2)
    plot_routing_hierarchy(hw_config, filename="test_results/routing_hierarchy.pdf")


def test_clustering_visualization():
    # A. Generate a mock graph (Stochastic Block Model)
    num_nodes = 300
    g = gt.Graph(directed=False)
    g.add_vertex(num_nodes)

    # B. Create Ground Truth (3 distinct clusters)
    initial_clusters = np.repeat([0, 1, 2], 100)

    # Add edges: dense inside clusters, sparse between them
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = 0.15 if initial_clusters[i] == initial_clusters[j] else 0.005
            if np.random.rand() < prob:
                g.add_edge(i, j)

    # C. Create Inferred Clustering (simulate some mistakes)
    inferred_clusters = initial_clusters.copy()
    # Scramble 10% of the nodes to simulate an imperfect clustering algorithm
    scramble_mask = np.random.rand(num_nodes) < 0.10
    inferred_clusters[scramble_mask] = np.random.choice([0, 1, 2], size=np.sum(scramble_mask))

    # D. Run visualization
    plot_clustering_comparison(
        g,
        initial_assignment=initial_clusters,
        inferred_assignment=inferred_clusters,
        filename="test_results/clustering_evaluation.pdf"
    )