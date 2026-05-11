from typing import Any
import numpy as np
import graph_tool.all as gt
from scipy.optimize import quadratic_assignment
from pydantic import BaseModel, ConfigDict

from netochi.mapping.mcmc.likelihood_state import MappingState
from netochi.mapping.interfaces import BaseMapper, MosaicMappingState
from netochi.input_generator.interfaces import MosaicMappingInput


class QAPMapper(BaseModel, BaseMapper[MosaicMappingState, MosaicMappingInput[Any]]):
    """
    Mapper based on the Quadratic Assignment Problem (QAP).
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)

    def run(self, mapping_input: MosaicMappingInput[Any]) -> MosaicMappingState:
        """Find core and address allocations using QAP FAQ heuristic."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        
        # Use MappingState for internal precomputations and intermediate storage
        # (It provides in_edges and core_distance logic)
        calc_state = MappingState(graph=graph, config=hw)
        
        # Number of actual nodes
        N_graph = graph.num_vertices()
        
        # Total hardware slots M
        M = hw.total_cores * hw.neurons_per_core
        
        # QAP requires A and B to be square and of the same size.
        # Size will be max(N_graph, M).
        size = max(N_graph, M)
        
        # 1. Build Adjacency Matrix A
        A = np.zeros((size, size))
        adj = gt.adjacency(graph).toarray()
        A[:N_graph, :N_graph] = adj
        
        # 2. Build Affinity Matrix B
        # B[k, l] is the affinity from slot k to slot l.
        B = np.zeros((size, size))
        
        for k in range(M):
            c_k = k // hw.neurons_per_core
            for l in range(M):
                c_l = l // hw.neurons_per_core
                
                dist = hw.core_distance(c_l, c_k)
                if dist == 0:
                    B[k, l] = 1.0
                else:
                    s_d = hw.num_slices_at_distance(dist)
                    B[k, l] = 1.0 / s_d
                    
        # 3. Solve QAP using FAQ
        # maximize=True because we want to maximize the sum of valid expected edges
        res = quadratic_assignment(A, B, method='faq', options={'maximize': True})
        
        # res.col_ind gives the assignment: node i goes to slot res.col_ind[i]
        perm = res.col_ind
        
        # 4. Map the results back to the arrays
        for node in range(N_graph):
            slot = perm[node]
            if slot >= M:
                # If node mapped to a dummy slot > M, wrap around
                slot = slot % M
                
            calc_state.c[node] = slot // hw.neurons_per_core
            calc_state.x[node] = slot % hw.neurons_per_core
            
        # 5. Greedy Slice Selection
        for tgt in range(calc_state.N):
            tgt_core = calc_state.c[tgt]
            for d in range(1, hw.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = hw.num_slices_at_distance(d)
                
                for s in range(n_slices):
                    count = 0
                    start, end = hw.get_slice_bounds(d, s)
                    
                    # calc_state._in_edges is a PrivateAttr, but we can access it for performance
                    # or use a helper if we want to be strict. Here we use the private attr.
                    for src in calc_state._in_edges[tgt]:
                        src_core = calc_state.c[src]
                        if hw.core_distance(tgt_core, src_core) == d:
                            if start <= calc_state.x[src] < end:
                                count += 1
                                
                    if count > max_sources:
                        max_sources = count
                        best_slice = s
                        
                calc_state.s[tgt, d] = best_slice
                
        return MosaicMappingState(
            mapping_input=mapping_input,
            neuron_core_idxs_assignment=calc_state.c,
            neuron_local_idxs_assignment=calc_state.x,
            neuron_slice_assignments=calc_state.s,
        )
