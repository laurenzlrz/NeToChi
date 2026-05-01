import numpy as np
import graph_tool.all as gt
from scipy.optimize import quadratic_assignment
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.core import BaseMapper, IFixedHardwareMapper, FixedHardwareInput

class QAPMapper(BaseMapper, IFixedHardwareMapper):
    """Mapper based on the Quadratic Assignment Problem (QAP)."""
    def map_fixed_hardware(self, mapping_input: FixedHardwareInput) -> MappingState:
        """Find core and address allocations using QAP FAQ heuristic."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        # Number of actual nodes
        N_graph = mapping_input.graph.num_vertices()
        
        # Total hardware slots M
        M = mapping_input.hw_config.total_cores * mapping_input.hw_config.neurons_per_core
        
        # QAP requires A and B to be square and of the same size.
        # Size will be max(N_graph, M).
        size = max(N_graph, M)
        
        # 1. Build Adjacency Matrix A
        A = np.zeros((size, size))
        adj = gt.adjacency(mapping_input.graph).toarray()
        A[:N_graph, :N_graph] = adj
        
        # 2. Build Affinity Matrix B
        # B[k, l] is the affinity from slot k to slot l.
        B = np.zeros((size, size))
        
        for k in range(size):
            if k >= M: continue # Dummy slot, no affinity
            c_k = k // mapping_input.hw_config.neurons_per_core
            
            for l in range(size):
                if l >= M: continue # Dummy slot, no affinity
                c_l = l // mapping_input.hw_config.neurons_per_core
                
                dist = mapping_input.hw_config.core_distance(c_l, c_k)
                if dist == 0:
                    B[k, l] = 1.0
                else:
                    s_d = mapping_input.hw_config.num_slices_at_distance(dist)
                    B[k, l] = 1.0 / s_d
                    
        # 3. Solve QAP using FAQ
        # maximize=True because we want to maximize the sum of valid expected edges
        res = quadratic_assignment(A, B, method='faq', options={'maximize': True})
        
        # res.col_ind gives the assignment: node i goes to slot res.col_ind[i]
        perm = res.col_ind
        
        # 4. Map the results back to the state (c and x)
        for node in range(N_graph):
            slot = perm[node]
            if slot >= M:
                # If node mapped to a dummy slot > M, wrap around or assign to slot 0
                # Assuming the hardware is large enough, this shouldn't happen for valid N_graph <= M
                slot = slot % M
                
            state.c[node] = slot // mapping_input.hw_config.neurons_per_core
            state.x[node] = slot % mapping_input.hw_config.neurons_per_core
            
        # 5. Greedy Slice Selection (s)
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
