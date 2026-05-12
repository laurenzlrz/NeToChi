from typing import Any
import graph_tool.all as gt
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicMappingInput


class HybridMapper(BaseModel, BaseMapper[MosaicNetworkMappingState[Any], MosaicMappingInput[Any]]):
    """
    Heuristic mapper combining initial greedy clustering with random refinements.
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)
    greedy_iterations: int = Field(default=10)

    def run(self, mapping_input: MosaicMappingInput[Any]) -> MosaicNetworkMappingState[Any]:
        """Execute hybrid mapping strategy."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        num_neurons = graph.num_vertices()
        
        # Initialize result state
        state: MosaicNetworkMappingState[Any] = MosaicNetworkMappingState.from_input(mapping_input)
        c_assignment = state.neuron_core_idxs_assignment
        x_assignment = state.neuron_local_idxs_assignment
        s_assignment = state.neuron_slice_assignments
        
        # 1. Run hSBM
        nested_state = gt.minimize_nested_blockmodel_dl(graph)
        blocks = nested_state.get_levels()[0].get_blocks().get_array()
        
        block_to_nodes = defaultdict(list)
        for v, b in enumerate(blocks):
            block_to_nodes[b].append(v)
            
        current_core = 0
        current_count = 0
        for b in sorted(block_to_nodes.keys()):
            for node in block_to_nodes[b]:
                if current_count >= hw.neurons_per_core:
                    current_core += 1
                    current_count = 0
                if current_core >= hw.total_cores:
                    current_core = hw.total_cores - 1
                c_assignment[node] = current_core
                current_count += 1
                
        # 2a. Local Address Assignment (x)
        adj = gt.adjacency(graph).toarray()
        core_to_nodes = defaultdict(list)
        for v in range(num_neurons):
            core_to_nodes[c_assignment[v]].append(v)
            
        for core, nodes in core_to_nodes.items():
            if not nodes: continue
            if len(nodes) == 1:
                x_assignment[nodes[0]] = 0
                continue
                
            features = adj[nodes, :]
            if np.all(features == features[0]):
                scores = np.arange(len(nodes))
            else:
                pca = PCA(n_components=1)
                scores = pca.fit_transform(features)[:, 0]
                
            sorted_indices = np.argsort(scores)
            for i, idx in enumerate(sorted_indices):
                x_assignment[nodes[idx]] = i
                
        # 2b. Greedy Slice Selection (s)
        for tgt in range(num_neurons):
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
