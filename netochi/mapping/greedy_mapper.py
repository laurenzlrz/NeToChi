from typing import Dict, Any, Optional
import graph_tool.all as gt
import numpy as np
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import BaseMapper, MosaicMappingState

class GreedyMapper(BaseMapper[MosaicMappingState, MosaicMappingInput[Any]]):
    """Mapper implementing a pure greedy heuristic based on node degrees and local clustering."""

    def run(self, mapping_input: MosaicMappingInput[Any]) -> MosaicMappingState:
        """
        Execute the greedy mapping algorithm.
        
        1. Nodes are ordered by total degree.
        2. Nodes are placed in cores that already contain their neighbors.
        3. Listening slices are optimized level-by-level to maximize valid Fan-In.
        """
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        num_neurons = graph.num_vertices()
        
        # 1. Initialize result arrays
        core_assignments = np.full(num_neurons, -1, dtype=np.int64)
        local_assignments = np.full(num_neurons, -1, dtype=np.int64)
        slice_assignments = np.zeros((num_neurons, hw.router_levels + 1), dtype=np.int64)
        
        # Track current core capacity
        core_occupancy = np.zeros(hw.total_cores, dtype=np.int64)
        
        # 2. Node Ordering: Sort by total degree (in + out) descending
        in_degrees = graph.get_in_degrees(graph.get_vertices())
        out_degrees = graph.get_out_degrees(graph.get_vertices())
        total_degrees = in_degrees + out_degrees
        sorted_nodes = np.argsort(total_degrees)[::-1]
        
        # 3. Phase I: Core and Local Index Assignment
        for node in sorted_nodes:
            core_scores = np.zeros(hw.total_cores)
            
            # Score cores based on where neighbors are already placed
            for neighbor in graph.get_all_neighbors(node):
                neighbor_core = core_assignments[int(neighbor)]
                if neighbor_core != -1:
                    core_scores[neighbor_core] += 1
            
            # Find the best available core (highest neighbor count, has space)
            best_cores_sorted = np.argsort(core_scores)[::-1]
            assigned = False
            for core_idx in best_cores_sorted:
                if core_occupancy[core_idx] < hw.neurons_per_core:
                    core_assignments[node] = core_idx
                    local_assignments[node] = core_occupancy[core_idx]
                    core_occupancy[core_idx] += 1
                    assigned = True
                    break
            
            if not assigned:
                raise RuntimeError(
                    f"Hardware Capacity Exceeded: Could not assign node {node} "
                    f"to any core (Total neurons: {num_neurons}, Hardware Capacity: {hw.total_neurons})"
                )

        # 4. Phase II: Greedy Slice Selection (Fan-In Optimization)
        for target_node in range(num_neurons):
            target_core = core_assignments[target_node]
            
            # Optimize slices for each hierarchy distance d
            for d in range(1, hw.router_levels + 1):
                num_slices = hw.num_slices_at_distance(d)
                best_slice = 0
                max_valid_edges = -1
                
                # Check each possible slice index s for this target neuron and distance
                for s_idx in range(num_slices):
                    valid_count = 0
                    start, end = hw.get_slice_bounds(d, s_idx)
                    
                    # Check incoming neighbors that are at distance d
                    for source_node in graph.get_in_neighbors(target_node):
                        source_core = core_assignments[int(source_node)]
                        if hw.core_distance(target_core, source_core) == d:
                            source_local = local_assignments[int(source_node)]
                            # If the source's local address fits in this slice, the edge is valid
                            if start <= source_local < end:
                                valid_count += 1
                    
                    # Keep track of the slice that maximizes valid incoming edges
                    if valid_count > max_valid_edges:
                        max_valid_edges = valid_count
                        best_slice = s_idx
                
                slice_assignments[target_node, d] = best_slice
                
        # 5. Build and return the validated MosaicMappingState
        return MosaicMappingState(
            mapping_input=mapping_input,
            neuron_core_idxs_assignment=core_assignments,
            neuron_local_idxs_assignment=local_assignments,
            neuron_slice_assignments=slice_assignments
        )
