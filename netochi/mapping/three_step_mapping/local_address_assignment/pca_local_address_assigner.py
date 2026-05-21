from collections import defaultdict
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA

from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput

import graph_tool.all as gt


class PcaLocalAddressAssigner(LocalAddressAssigner):


    def assign_addresses(self, graph: gt.Graph, clustering: ClusterAndHwOutput) -> Dict[int, int]:
        """
        infer neuron_id -> local_idx using PCA
        """
        num_neurons = graph.num_vertices()
        adj = gt.adjacency(graph).toarray()
        core_to_nodes = defaultdict(list)

        neuron_local_assignment = {}

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

        return neuron_local_assignment