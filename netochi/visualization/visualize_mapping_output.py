

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMosaicMappingState

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import graph_tool.all as gt
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass



def plot_hardware_mapping(
        graph: gt.Graph,
        state: BaseMosaicMappingState,
        config: MosaicHardwareConfig,
        filename: str = "hardware_mapping_circular_updated.pdf"
):
    """
    Plots the hardware routing hierarchy and mapped neural network.
    Cores are circles. Neurons are positioned within slice wedges.
    Radial lines demarcate slice boundaries for the maximum routing distance.
    Neuron angles are offset by ~1/2 the core-to-core angle.
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    num_cores = config.total_cores
    max_level = config.router_levels

    # ==========================================
    # 2.1 Routing Hierarchy Coordinates
    # ==========================================
    core_radius = 2.0
    core_spacing = max(6.0, core_radius * 3.0)
    core_pos = {i: (i * core_spacing, 0) for i in range(num_cores)}

    # Calculate Router positions level by level
    nodes_by_level = {0: [(i, core_pos[i]) for i in range(num_cores)]}

    for level in range(1, max_level + 1):
        nodes_by_level[level] = []
        children = nodes_by_level[level - 1]

        for i in range(0, len(children), config.nodes_per_router):
            group = children[i: i + config.nodes_per_router]
            avg_x = sum(pos[0] for _, pos in group) / len(group)
            r_pos = (avg_x, core_radius + level * 4.0)
            nodes_by_level[level].append((f"R{level}_{i}", r_pos))

            for child_id, child_pos in group:
                target_y = child_pos[1] + core_radius if level == 1 else child_pos[1]
                ax.plot([r_pos[0], child_pos[0]], [r_pos[1], target_y],
                        color='gray', lw=2, zorder=0)

            ax.scatter(*r_pos, marker='s', color='white', s=600, edgecolor='black', zorder=2)
            ax.text(r_pos[0], r_pos[1], f"R{level}", ha='center', va='center', fontsize=10, fontweight='bold')

    # ==========================================
    # 2.2 Draw Cores and Slice Boundary Lines
    # ==========================================
    # Get the number of slices at the maximum routing level
    max_slices = config.num_slices_at_distance(config.max_distance)
    slice_width_rad = (2 * np.pi) / max_slices

    for i in range(num_cores):
        cx, cy = core_pos[i]

        # Draw the Core Circle
        circle = Circle((cx, cy), core_radius, fill=True, facecolor='#f0f8ff',
                        edgecolor='dodgerblue', linestyle='-', lw=2, zorder=1)
        ax.add_patch(circle)

        # Label the core
        ax.text(cx, cy - core_radius - 0.7, f"C{i}", ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black"))

        # Draw radial lines to visualize the slice boundaries
        for s in range(max_slices):
            # Boundaries are drawn from 0 to 2pi
            angle = 2 * np.pi * (s / max_slices) + np.pi/2 # neg. offset because angle = 0 is horizontal line and it wanders counter-clockwise
            dx = core_radius * np.cos(angle)
            dy = core_radius * np.sin(angle)

            ax.plot([cx, cx + dx], [cy, cy + dy],
                    color='dodgerblue', linestyle='--', lw=1.5, zorder=1, alpha=0.8)

    # ==========================================
    # 2.3 Calculate Neuron Positions WITHIN Slices and Offset Angle
    # ==========================================
    neuron_pos = {}
    neuron_radius = core_radius * 0.75  # Keep nodes slightly inside boundary

    # Calculate the base angle between core centers to use for the offset
    base_angle_shift = - np.pi / config.neurons_per_core + np.pi / 2 # neg. offset because angle = 0 is horizontal line and it wanders counter-clockwise

    for v in graph.vertices():
        u = int(v)
        c = state.neuron_core_idxs_assignment[u]
        loc = state.neuron_local_idxs_assignment[u]

        cx, cy = core_pos[c]

        # 1. First, calculate which slice this neuron is located in
        # (Based on its local index and number of neurons)
        neuron_slice_idx = int((loc / config.neurons_per_core) * max_slices)

        # 2. Calculate the base angle of that slice's starting boundary
        slice_start_angle = - 2 * np.pi * (neuron_slice_idx / max_slices)

        # 3. Apply the global angular shift requested ("by 1/2 angle between cores")
        shifted_start_angle = slice_start_angle + base_angle_shift

        # 4. Map the local index smoothly WITHIN that slice wedge.
        # This keeps them visually confined to their slice while sorting by local index.
        normalized_loc_in_slice = (loc % (config.neurons_per_core // max_slices)) / (
                    config.neurons_per_core // max_slices)
        angle_within_slice = - normalized_loc_in_slice * slice_width_rad

        final_angle = shifted_start_angle + angle_within_slice

        nx = cx + neuron_radius * np.cos(final_angle)
        ny = cy + neuron_radius * np.sin(final_angle)
        neuron_pos[u] = (nx, ny)

    # ==========================================
    # 2.4 Draw Graph Edges (Valid vs Invalid)
    # ==========================================
    for e in graph.edges():
        u = int(e.source());
        v = int(e.target())
        c_u = state.neuron_core_idxs_assignment[u];
        c_v = state.neuron_core_idxs_assignment[v]
        x_u = state.neuron_local_idxs_assignment[u];
        dist = config.core_distance(c_u, c_v)

        if dist == 0:
            is_valid = True
        else:
            target_slice_idx = state.neuron_slice_assignments[v, dist]
            is_valid = config.is_valid_connection(c_u, c_v, x_u, target_slice_idx)

        color = 'mediumseagreen' if is_valid else 'crimson'
        alpha = 0.5 if is_valid else 0.8;
        zorder = 2 if is_valid else 3
        p1 = neuron_pos[u];
        p2 = neuron_pos[v]
        arc_rad = 0.4 if c_u == c_v else 0.2

        arrow = FancyArrowPatch(
            p1, p2, connectionstyle=f"arc3,rad={arc_rad}",
            color=color, alpha=alpha, arrowstyle='-|>',
            mutation_scale=10, lw=1.2, zorder=zorder
        )
        ax.add_patch(arrow)

    # ==========================================
    # 2.5 Scatter the Neurons
    # ==========================================
    n_x = [pos[0] for pos in neuron_pos.values()];
    n_y = [pos[1] for pos in neuron_pos.values()]
    ax.scatter(n_x, n_y, color='black', s=25, zorder=4, edgecolor='white', lw=0.5)

    ax.set_title(
        f"Mapping Output Validation\n"
        f"Green = Valid Hardware Edge | Red = Inconsistent Map\n",
        pad=20, fontsize=15
    )
    ax.autoscale()
    ax.axis('equal');
    ax.axis('off')
    plt.margins(0.1);
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"Radial validation plot successfully saved to {filename}")


# ==========================================
# 1. Mock Classes (Based on your definitions)
# ==========================================
class MosaicHardwareConfig:
    def __init__(self, nodes_per_router: int, neurons_per_core: int, router_levels: int, slice_factor: int = 2):
        self.nodes_per_router = nodes_per_router
        self.neurons_per_core = neurons_per_core
        self.router_levels = router_levels
        self.slice_factor = slice_factor
        self.total_cores = int(nodes_per_router ** router_levels)
        self.total_neurons = int(self.total_cores * neurons_per_core)
        self.max_distance = router_levels

    def core_distance(self, core_a: int, core_b: int) -> int:
        if core_a == core_b: return 0
        base = self.nodes_per_router
        for level in range(self.router_levels - 1, -1, -1):
            if (core_a // (base ** level)) % base != (core_b // (base ** level)) % base:
                return level + 1
        return 0

    def num_slices_at_distance(self, distance: int) -> int:
        return int(min(self.slice_factor ** distance, self.neurons_per_core))

    def get_slice_bounds(self, distance: int, slice_idx: int) -> tuple[int, int]:
        slices = self.num_slices_at_distance(distance)
        start = (slice_idx * self.neurons_per_core) // slices
        end = ((slice_idx + 1) * self.neurons_per_core) // slices
        return start, end

    def is_valid_connection(self, source_core: int, target_core: int, source_local_addr: int,
                            target_slice_idx: int) -> bool:
        dist = self.core_distance(target_core, source_core)
        if dist == 0: return True
        start, end = self.get_slice_bounds(dist, target_slice_idx)
        return start <= source_local_addr < end


@dataclass
class MockMappingState:
    neuron_core_idxs_assignment: np.ndarray
    neuron_local_idxs_assignment: np.ndarray
    neuron_slice_assignments: np.ndarray


# ==========================================
# 2. Paste the plot_hardware_mapping function here
# (Omitted for brevity, paste the function from the previous response)
# ==========================================
# def plot_hardware_mapping(graph, state, config, filename="hardware_mapping_validation.pdf"): ...

# ==========================================
# 3. Example Execution
# ==========================================
if __name__ == "__main__":
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

    state = MockMappingState(
        neuron_core_idxs_assignment=cores,
        neuron_local_idxs_assignment=locals_idx,
        neuron_slice_assignments=slices
    )

    # D. Generate the plot
    plot_hardware_mapping(g, state, config, filename="../results/example_hw_mapping.pdf")
