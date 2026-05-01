import graph_tool.all as gt
import numpy as np
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.core import BaseMapper, IFixedHardwareMapper, FixedHardwareInput

class GreedyMapper(BaseMapper, IFixedHardwareMapper):
    """Mapper implementing a pure greedy heuristic based on node degrees."""
    def map_fixed_hardware(self, mapping_input: FixedHardwareInput) -> MappingState:
        """Map nodes greedily to cores with available capacity."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        
        # Get degrees
        in_degrees = mapping_input.graph.get_in_degrees(mapping_input.graph.get_vertices())
        out_degrees = mapping_input.graph.get_out_degrees(mapping_input.graph.get_vertices())
        total_degrees = in_degrees + out_degrees
        
        # Sort nodes by degree descending
        sorted_nodes = np.argsort(total_degrees)[::-1]
        
        core_counts = np.zeros(mapping_input.hw_config.total_cores, dtype=int)
        
        # Initialize cores to -1 (unassigned)
        state.c.fill(-1)
        
        for node in sorted_nodes:
            core_scores = np.zeros(mapping_input.hw_config.total_cores)
            
            # Reward placing this node in the same core as its already-placed neighbors
            for neighbor in mapping_input.graph.get_all_neighbors(node):
                neighbor_core = state.c[neighbor]
                if neighbor_core != -1:
                    core_scores[neighbor_core] += 1
            
            # Find best core with available capacity
            best_cores = np.argsort(core_scores)[::-1]
            for core in best_cores:
                if core_counts[core] < mapping_input.hw_config.neurons_per_core:
                    state.c[node] = core
                    state.x[node] = core_counts[core]
                    core_counts[core] += 1
                    break
                    
        # Greedy Slice Selection (s)
        for tgt in range(state.N):
            tgt_core = state.c[tgt]
            for d in range(1, mapping_input.hw_config.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = mapping_input.hw_config.num_slices_at_distance(d)
                
                for s in range(n_slices):
                    count = 0
                    start, end = mapping_input.hw_config.get_slice_bounds(d, s)
                    
                    for src in state.in_edges[tgt]:
                        src = int(src)
                        src_core = state.c[src]
                        if mapping_input.hw_config.core_distance(tgt_core, src_core) == d:
                            if start <= state.x[src] < end:
                                count += 1
                                
                    if count > max_sources:
                        max_sources = count
                        best_slice = s
                        
                state.s[tgt, d] = best_slice
                
        return state
