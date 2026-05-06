import graph_tool.all as gt
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from netochi.mapping.hardware_config import HardwareConfig
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.core import FixedHardwareInput

class HybridMapper:
    """Mapper combining hSBM clustering and PCA-based local assignment."""
    
    def get_name(self) -> str:
        return self.__class__.__name__

    def map_fixed_hardware(self, mapping_input: FixedHardwareInput) -> MappingState:
        """Map nodes using hSBM for cores and PCA for local addresses."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        # 1. Run hSBM
        nested_state = gt.minimize_nested_blockmodel_dl(mapping_input.graph)
        blocks = nested_state.get_levels()[0].get_blocks().get_array()
        
        # Map blocks to hardware cores respecting capacity constraint n
        block_to_nodes = defaultdict(list)
        for v, b in enumerate(blocks):
            block_to_nodes[b].append(v)
            
        current_core = 0
        current_count = 0
        for b in sorted(block_to_nodes.keys()):
            for node in block_to_nodes[b]:
                if current_count >= mapping_input.hw_config.neurons_per_core:
                    current_core += 1
                    current_count = 0
                if current_core >= mapping_input.hw_config.total_cores:
                    # In case the graph is larger than hardware capacity, wrap around or ignore.
                    # Assuming graph fits in hardware:
                    current_core = mapping_input.hw_config.total_cores - 1
                state.c[node] = current_core
                current_count += 1
                
        # 2a. Local Address Assignment (x)
        adj = gt.adjacency(mapping_input.graph).toarray()
        core_to_nodes = defaultdict(list)
        for v in range(state.N):
            core_to_nodes[state.c[v]].append(v)
            
        for core, nodes in core_to_nodes.items():
            if len(nodes) == 0:
                continue
            elif len(nodes) == 1:
                state.x[nodes[0]] = 0
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
                state.x[nodes[idx]] = i
                
        # 2b. Greedy Slice Selection (s)
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
