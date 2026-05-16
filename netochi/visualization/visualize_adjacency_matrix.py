import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt

from dataclasses import dataclass

from netochi.mapping.interfaces import BaseMosaicMappingState



def plot_sorted_adjacency(graph: gt.Graph, state: BaseMosaicMappingState, filename: str):
    """
    Plots the adjacency matrix of a graph, sorted by core assignment and then local index.
    Draws red boundary lines to separate different cores and labels the axes with Core IDs.
    """
    c = state.neuron_core_idxs_assignment
    x = state.neuron_local_idxs_assignment

    # Sort primarily by core ID (c), secondarily by local ID (x)
    sorted_indices = np.lexsort((x, c))
    adj_matrix = gt.adjacency(graph)
    adj_sorted = adj_matrix[sorted_indices, :][:, sorted_indices]

    plt.figure(figsize=(8, 8))
    plt.spy(adj_sorted, markersize=4, color='black')

    sorted_cores = c[sorted_indices]
    core_boundaries = np.where(np.diff(sorted_cores) != 0)[0] + 1

    # Draw red boundary lines
    for boundary in core_boundaries:
        plt.axhline(boundary - 0.5, color='red', linewidth=1.5, alpha=0.6)
        plt.axvline(boundary - 0.5, color='red', linewidth=1.5, alpha=0.6)

    # --- NEW: Calculate and set core labels on axes ---
    tick_positions = []
    tick_labels = []

    start_idx = 0
    # Append total length to boundaries to handle the last core block easily
    all_boundaries = np.append(core_boundaries, len(sorted_cores))

    for boundary in all_boundaries:
        # Calculate the midpoint of the core block to center the label
        midpoint = (start_idx + boundary - 1) / 2.0
        tick_positions.append(midpoint)

        # Grab the actual core ID from the sorted array
        core_id = sorted_cores[start_idx]
        tick_labels.append(f"C{core_id}")

        start_idx = boundary

    # Apply the custom ticks
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)

    # Remove the standard tick marks (the little lines) for a cleaner look
    plt.tick_params(axis='both', which='both', length=0)
    # ---------------------------------------------------

    plt.title("Adjacency Matrix\n(Neurons Grouped by Cores)", pad=20)
    plt.xlabel("Target Neuron")
    plt.ylabel("Source Neuron")

    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


# ==========================================
# Example Usage
# ==========================================

# 1. Mocking the Mapping State for the example
@dataclass
class MockState:
    neuron_core_idxs_assignment: np.ndarray
    neuron_local_idxs_assignment: np.ndarray


if __name__ == "__main__":
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

                prob = 0.70 if core_i == core_j else 0.05
                if np.random.rand() < prob:
                    g.add_edge(i, j)

    # C. Create assignments simulating a perfect mapping algorithm
    # To prove the sorting works, we will scramble the neurons before passing them in
    shuffled_indices = np.random.permutation(total_neurons)

    core_assignments = np.zeros(total_neurons, dtype=int)
    local_assignments = np.zeros(total_neurons, dtype=int)

    for real_idx, original_idx in enumerate(shuffled_indices):
        core_assignments[original_idx] = real_idx // neurons_per_core
        local_assignments[original_idx] = real_idx % neurons_per_core

    mock_state = MockState(
        neuron_core_idxs_assignment=core_assignments,
        neuron_local_idxs_assignment=local_assignments
    )

    # D. Run the visualization!
    # Because our mock mapping matches the dense cluster generation,
    # you should see 3 dense black squares separated by red lines.
    plot_sorted_adjacency(g, mock_state, filename="../results/adjacency_matrix.pdf")