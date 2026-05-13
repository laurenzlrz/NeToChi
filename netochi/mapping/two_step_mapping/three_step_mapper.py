from netochi.mapping.two_step_mapping.interfaces import HierarchicalClusterer, LocalAddressAssigner, SliceAssigner

import graph_tool.all as gt
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from netochi.mapping.two_step_mapping.interfaces import HierarchicalClusterOutput

from netochi.mapping.interfaces import MosaicHWMappingState
from netochi.input_generator.interfaces import MappingInput



class ThreeStepMapper:

    def __init__(self, clusterer: HierarchicalClusterer, address_assigner: LocalAddressAssigner, slice_assigner: SliceAssigner):
        self.clusterer = clusterer
        self.address_assigner = address_assigner
        self.slice_assigner = slice_assigner


    def run(self, mapping_input: MappingInput) -> MosaicHWMappingState:
        graph = mapping_input.graph

        # --- 1. clustering ---
        clustering: HierarchicalClusterOutput = self.clusterer.cluster(input_data=mapping_input)

        # --- 2. Local address assignment ---
        num_neurons = graph.num_vertices()
        neuron_local_assignment = {}
        adj = gt.adjacency(graph).toarray()
        core_to_nodes = defaultdict(list)
        for v in range(num_neurons):
            core_to_nodes[clustering.cluster_assignment[v]].append(v)

        for core, nodes in core_to_nodes.items():
            if not nodes: continue
            if len(nodes) == 1:
                neuron_local_assignment[nodes[0]] = 0
                continue

            features = adj[nodes, :]
            if np.all(features == features[0]):
                scores = np.arange(len(nodes))
            else:
                pca = PCA(n_components=1)
                scores = pca.fit_transform(features)[:, 0]

            sorted_indices = np.argsort(scores)
            for i, idx in enumerate(sorted_indices):
                neuron_local_assignment[nodes[idx]] = i

        # --- 3. Slice assignment ---
        neuron_slice_assignment = self.slice_assigner.assign_slices(clustering=clustering, graph=graph)

        # todo add other attributes to Mosaic hw mapping state
        state = MosaicHWMappingState()
        state.neuron_slice_assignments = neuron_slice_assignment
        state.neuron_local_idxs_assignment = neuron_local_assignment
        state.neuron_core_idxs_assignment = clustering.cluster_assignment
        return state