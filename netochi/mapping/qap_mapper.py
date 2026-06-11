from typing import Any
import numpy as np
import graph_tool.all as gt
from scipy.optimize import quadratic_assignment
from pydantic import BaseModel, ConfigDict

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment


class QAPMapper(BaseModel, BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper based on the Quadratic Assignment Problem (QAP).
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        """Find core and address allocations using QAP FAQ heuristic."""
        graph = mapping_input.graph
        gt_hw = mapping_input.hw_config
        N_graph = graph.num_vertices()
        
        # Initialize result state
        # state = MosaicNetworkMappingState.from_input(mapping_input)
        assignment = MosaicAssignment.zero(num_neurons=N_graph, hw=gt_hw)
        c_assignment = assignment.neuron_core_pre_assignment
        x_assignment = assignment.neuron_idx_pre_assignment
        s_assignment = assignment.neuron_slice_assignment

        size = gt_hw.total_neurons

        # 1. Build Adjacency Matrix A
        A = np.zeros((size, size))
        adj = gt.adjacency(graph).toarray()
        A[:N_graph, :N_graph] = adj
        
        # 2. Build Affinity Matrix B
        B = np.zeros((size, size))
        for k in range(N_graph):
            c_k = k // gt_hw.neurons_per_core
            for l in range(N_graph):
                c_l = l // gt_hw.neurons_per_core
                dist = gt_hw.core_distance(c_l, c_k)
                if dist == 0:
                    B[k, l] = 1.0
                else:
                    s_d = gt_hw.num_slices_at_distance(dist)
                    B[k, l] = 1.0 / s_d
                    
        # 3. Solve QAP using FAQ
        res = quadratic_assignment(A, B, method='faq', options={'maximize': True})
        perm = res.col_ind
        
        # 4. Map the results back
        for node in range(N_graph):
            slot = perm[node]
            if slot >= M:
                slot = slot % M
            c_assignment[node] = slot // gt_hw.neurons_per_core
            x_assignment[node] = slot % gt_hw.neurons_per_core
            
        # 5. Greedy Slice Selection
        for tgt in range(N_graph):
            tgt_core = c_assignment[tgt]
            for d in range(1, gt_hw.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = gt_hw.num_slices_at_distance(d)
                for s_idx in range(n_slices):
                    count = 0
                    start, end = gt_hw.get_slice_bounds(d, s_idx)
                    for src in graph.get_in_neighbors(tgt):
                        src_core = c_assignment[src]
                        if gt_hw.core_distance(tgt_core, src_core) == d:
                            if start <= x_assignment[src] < end:
                                count += 1
                    if count > max_sources:
                        max_sources = count
                        best_slice = s_idx
                s_assignment[tgt, d] = best_slice
                
        return state
