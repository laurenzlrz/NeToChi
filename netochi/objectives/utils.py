from typing import Dict, Any

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMosaicMappingState



def compute_e_valid(state: BaseMosaicMappingState[Any], data: Dict[str, Any]) -> int:
    """
    data needs to have:
    - N: num nodes
    - in_edges for every target node
    """
    hw = state.hw_to_evaluate
    c, x, s = state.c, state.x, state.s
    e_valid = 0
    for tgt in range(data['N']):
        c_tgt = c[tgt]
        for src in data['in_edges'][tgt]:
            c_src = c[src]
            dist = hw.core_distance(int(c_tgt), int(c_src))
            if dist == 0:
                e_valid += 1
            else:
                if hw.get_slice_bounds(dist, s[tgt][dist])[0] <= x[src] < hw.get_slice_bounds(dist, s[tgt][dist])[1]:
                    e_valid += 1
    return e_valid


def compute_inconsistencies(state: BaseMosaicMappingState[Any], data: Dict[str, Any]) -> int:
    """
    data needs to have:
    - N: num nodes
    - in_edges for every target node (adjacency list of incoming edges)
    """
    hw = state.hw_to_evaluate
    c, x, s = state.c, state.x, state.s
    inconsistencies = 0

    for tgt in range(data['N']):
        c_tgt = c[tgt]
        for src in data['in_edges'][tgt]:
            c_src = c[src]
            dist = hw.core_distance(int(c_tgt), int(c_src))

            # Distance 0 is always valid (fully connected), so we only check dist > 0
            if dist > 0:
                chosen_slice = s[tgt][dist]
                lower_bound, upper_bound = hw.get_slice_bounds(dist, chosen_slice)

                # If the source intra-core index is OUTSIDE the bounds, it's an inconsistency
                if not (lower_bound <= x[src] < upper_bound):
                    inconsistencies += 1

    return inconsistencies

def compute_total_hw_connections(hw_config: MosaicHardwareConfig) -> int:
    """Computes the total number of hardware connections on the chip."""
    R = hw_config.nodes_per_router
    N = hw_config.neurons_per_core

    connections_per_neuron = 0

    # --- Distance 0 (Intra-core Connections) ---
    # A neuron connects to all N neurons in its core (including itself)
    connections_per_neuron += N

    # --- Distance d > 0 (Inter-core Connections) ---
    for d in range(1, hw_config.router_levels + 1):
        # Number of peer cores exactly d hops away
        num_cores_at_dist_d = (R - 1) * (R ** (d - 1))
        start, end = hw_config.get_slice_bounds(d, 0)
        slice_size_d = end - start
        connections_per_neuron += num_cores_at_dist_d * slice_size_d

    return hw_config.total_neurons * connections_per_neuron