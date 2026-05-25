from typing import Any
import numpy as np
import graph_tool.all as gt
from scipy.optimize import quadratic_assignment
from pydantic import BaseModel, ConfigDict

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicHWMappingInput


class QAPMapper(BaseModel, BaseMapper[MosaicNetworkMappingState[Any], MosaicHWMappingInput[Any]]):
    """
    Mapper based on the Quadratic Assignment Problem (QAP).
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)

    def run(self, mapping_input: MosaicHWMappingInput[Any]) -> MosaicNetworkMappingState[Any]:
        """Find core and address allocations using QAP FAQ heuristic."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        N_graph = graph.num_vertices()
        
        # Initialize result state
        state = MosaicNetworkMappingState.from_input(mapping_input)
        c_assignment = state.neuron_core_idxs_assignment
        x_assignment = state.neuron_local_idxs_assignment
        s_assignment = state.neuron_slice_assignments
        
        M = hw.total_cores * hw.neurons_per_core
        size = max(N_graph, M)
        
        # 1. Build Adjacency Matrix A
        A = np.zeros((size, size))
        adj = gt.adjacency(graph).toarray()
        A[:N_graph, :N_graph] = adj
        
        # 2. Build Affinity Matrix B
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
        res = quadratic_assignment(A, B, method='faq', options={'maximize': True})
        perm = res.col_ind
        
        # 4. Map the results back
        for node in range(N_graph):
            slot = perm[node]
            if slot >= M:
                slot = slot % M
            c_assignment[node] = slot // hw.neurons_per_core
            x_assignment[node] = slot % hw.neurons_per_core
            
        # 5. Greedy Slice Selection
        for tgt in range(N_graph):
            tgt_core = c_assignment[tgt]
            for d in range(1, hw.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = hw.num_slices_at_distance(d)
                for s_idx in range(n_slices):
                    count = 0
                    start, end = hw.get_slice_bounds(d, s_idx)
                    for src in graph.get_in_neighbors(tgt):
                        src_core = c_assignment[src]
                        if hw.core_distance(tgt_core, src_core) == d:
                            if start <= x_assignment[src] < end:
                                count += 1
                    if count > max_sources:
                        max_sources = count
                        best_slice = s_idx
                s_assignment[tgt, d] = best_slice
                
        return state
