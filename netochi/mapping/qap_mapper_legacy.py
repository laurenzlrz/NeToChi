from typing import Any, cast
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
        assignment = MosaicAssignment.zero(num_neurons=N_graph, hw=gt_hw)
        c_assignment = assignment.neuron_core_pre_assignment
        x_assignment = assignment.neuron_idx_pre_assignment
        s_assignment = assignment.neuron_slice_assignment

        hw_num_neurons = gt_hw.total_neurons

        # 1. Build Adjacency Matrix A
        A = np.zeros((hw_num_neurons, hw_num_neurons))
        adj = cast(Any, gt.adjacency(graph)).toarray()
        A[:N_graph, :N_graph] = adj
        
        # 2. Build Affinity Matrix B
        B = np.zeros((hw_num_neurons, hw_num_neurons))
        for k in range(hw_num_neurons):
            c_k = gt_hw.global_neuron_to_local(k)[0]
            for l in range(hw_num_neurons):
                c_l = gt_hw.global_neuron_to_local(l)[0]
                dist = gt_hw.core_distance(int(c_l), int(c_k))
                s_d = gt_hw.num_slices_at_distance(dist)
                B[k, l] = 1.0 / s_d
                    
        # 3. Solve QAP using FAQ
        res = quadratic_assignment(A, B, method='faq', options={'maximize': True})
        perm = res.col_ind
        
        # 4. Map the results back
        for node in range(N_graph):
            slot = perm[node]
            local_tuple = gt_hw.global_neuron_to_local(slot)
            c_assignment[node] = local_tuple[0]
            x_assignment[node] = local_tuple[1]
            
        # 5. Greedy Slice Selection
        for tgt in range(N_graph):
            tgt_core = int(c_assignment[tgt])
            for d in range(1, gt_hw.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = gt_hw.num_slices_at_distance(d)
                for s_idx in range(n_slices):
                    count = 0
                    start, end = gt_hw.get_slice_bounds(d, s_idx)
                    for src in graph.get_in_neighbors(tgt):
                        src_core = int(c_assignment[int(src)])
                        if gt_hw.core_distance(tgt_core, src_core) == d:
                            if start <= x_assignment[int(src)] < end:
                                count += 1
                    if count > max_sources:
                        max_sources = count
                        best_slice = s_idx
                s_assignment[tgt, d] = best_slice
                
        return MosaicNetworkMappingState(
            _mapping_input=mapping_input,
            assignment=assignment
        )

