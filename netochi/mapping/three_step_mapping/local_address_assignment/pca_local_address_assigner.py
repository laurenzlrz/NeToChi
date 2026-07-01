from collections import defaultdict

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput

import graph_tool.all as gt


class PcaLocalAddressAssigner(LocalAddressAssigner):


    def assign_addresses(self, graph: gt.Graph, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_] :
        """
        infer neuron_id -> local_idx using PCA
        """
        num_neurons = graph.num_vertices()
        from typing import cast, Any
        adj = cast(Any, gt.adjacency(graph)).toarray()
        core_to_nodes = defaultdict(list)

        neuron_local_assignment = np.zeros(num_neurons, dtype=np.int_)

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

            sorted_relative_positions = np.argsort(scores)
            sorted_actual_nodes = np.array(nodes)[sorted_relative_positions]
            neuron_local_assignment[sorted_actual_nodes] = np.arange(len(nodes))

        return neuron_local_assignment

    def get_name(self) -> str:
        return "PCA"