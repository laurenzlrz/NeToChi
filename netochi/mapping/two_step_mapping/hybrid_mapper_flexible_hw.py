from typing import Any
import graph_tool.all as gt
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.two_step_mapping.clustering.hierarchical_community_detection.hcd_clusterer import HcdClusterer
from netochi.mapping.two_step_mapping.interfaces import HierarchicalClusterOutput
from netochi.mapping.two_step_mapping.slice_assignment.slice_assignment_utils import compute_best_slice_assignment, \
    compute_dists_between_cores, compute_core_sizes

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState, MosaicHWMappingState
from netochi.input_generator.interfaces import MosaicMappingInput, MappingInput


class HybridMapper(BaseModel, BaseMapper[MosaicNetworkMappingState[Any], MosaicMappingInput[Any]]):
    """
    Heuristic mapper combining initial greedy clustering with random refinements.
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)
    greedy_iterations: int = Field(default=10)

    def run(self, mapping_input: MappingInput[Any]) -> MosaicHWMappingState:
        """Execute hybrid mapping strategy."""
        graph = mapping_input.graph
        num_neurons = graph.num_vertices()

        # 1. Run hSBM
        clusterer = HcdClusterer()
        clustering: HierarchicalClusterOutput = clusterer.cluster(input_data=mapping_input)
        neuron_core_idxs_assignment = clustering.cluster_assignment

        # 2. Local Address Assignment (x)
        neuron_local_idx_assignment = {}
        adj = gt.adjacency(graph).toarray()
        core_to_nodes = defaultdict(list)
        for v in range(num_neurons):
            core_to_nodes[neuron_core_idxs_assignment[v]].append(v)
            
        for core, nodes in core_to_nodes.items():
            if not nodes: continue
            if len(nodes) == 1:
                neuron_local_idx_assignment[nodes[0]] = 0
                continue
                
            features = adj[nodes, :]
            if np.all(features == features[0]):
                scores = np.arange(len(nodes))
            else:
                pca = PCA(n_components=1)
                scores = pca.fit_transform(features)[:, 0]
                
            sorted_indices = np.argsort(scores)
            for i, idx in enumerate(sorted_indices):
                neuron_local_idx_assignment[nodes[idx]] = i
                
        # 3. Greedy Slice Selection (s)
        core_distances, max_distance = compute_dists_between_cores(cluster_output=clustering)
        core_sizes = compute_core_sizes(cluster_output=clustering)
        neuron_slice_assignments = compute_best_slice_assignment(num_neurons, core_assignment=neuron_core_idxs_assignment, max_distance=max_distance, core_sizes=core_sizes, core_distances=core_distances, graph=graph)

        # return
        state = MosaicHWMappingState()
        state.neuron_slice_assignments = neuron_slice_assignments
        state.neuron_local_idxs_assignment = neuron_local_idx_assignment
        state.neuron_core_idxs_assignment = neuron_core_idxs_assignment
        return state

