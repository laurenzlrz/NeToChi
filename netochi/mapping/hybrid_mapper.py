from typing import Any
import graph_tool.all as gt
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from pydantic import BaseModel, ConfigDict

from netochi.mapping.mcmc.likelihood_state import MappingState
from netochi.mapping.interfaces import BaseMapper, MosaicMappingState
from netochi.input_generator.interfaces import MosaicMappingInput


class HybridMapper(BaseModel, BaseMapper[MosaicMappingState, MosaicMappingInput[Any]]):
    """
    Mapper combining hSBM clustering and PCA-based local assignment.
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)

    def run(self, mapping_input: MosaicMappingInput[Any]) -> MosaicMappingState:
        """Map nodes using hSBM for cores and PCA for local addresses."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        
        calc_state = MappingState(graph=graph, config=hw)
        
        # 1. Run hSBM
        nested_state = gt.minimize_nested_blockmodel_dl(graph)
        blocks = nested_state.get_levels()[0].get_blocks().get_array()
        
        # Map blocks to hardware cores respecting capacity constraint neurons_per_core
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
                    # In case the graph is larger than hardware capacity, clamp to last core
                    current_core = hw.total_cores - 1
                calc_state.c[node] = current_core
                current_count += 1
                
        # 2a. Local Address Assignment (x)
        adj = gt.adjacency(graph).toarray()
        core_to_nodes = defaultdict(list)
        for v in range(calc_state.N):
            core_to_nodes[calc_state.c[v]].append(v)
            
        for core, nodes in core_to_nodes.items():
            if len(nodes) == 0:
                continue
            elif len(nodes) == 1:
                calc_state.x[nodes[0]] = 0
                continue
                
            features = adj[nodes, :]
            # Sort by 1st Principal Component to group similar out-connections
            if np.all(features == features[0]):
                scores = np.arange(len(nodes))
            else:
                pca = PCA(n_components=1)
                scores = pca.fit_transform(features)[:, 0]
                
            sorted_indices = np.argsort(scores)
            for i, idx in enumerate(sorted_indices):
                calc_state.x[nodes[idx]] = i
                
        # 2b. Greedy Slice Selection (s)
        for tgt in range(calc_state.N):
            tgt_core = calc_state.c[tgt]
            for d in range(1, hw.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = hw.num_slices_at_distance(d)
                
                for s in range(n_slices):
                    count = 0
                    start, end = hw.get_slice_bounds(d, s)
                    
                    # calc_state._in_edges is a PrivateAttr, access it for performance
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
