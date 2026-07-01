from typing import Any, cast
import graph_tool.all as gt
import numpy as np
from scipy.optimize import quadratic_assignment

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput, ClustererFixedHw


class QapHwClusterer(ClustererFixedHw):
    """
    Mapper based on the Quadratic Assignment Problem (QAP).

    Refactored to be a plain Python class.
    """


    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        """Find core and address allocations using QAP FAQ heuristic."""
        graph = input_data.graph
        gt_hw = input_data.hw_config
        N_graph = graph.num_vertices()


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
                dist = gt_hw.core_distance(c_l, c_k)
                s_d = gt_hw.num_slices_at_distance(dist)
                B[k, l] = 1.0 / s_d

        # 3. Solve QAP using FAQ
        res = quadratic_assignment(A, B, method='faq', options={'maximize': True})
        perm = res.col_ind

        core_assignment = np.zeros(N_graph, dtype=np.int_)
        num_clusters = gt_hw.total_cores

        # 4. Map the results back
        for node in range(N_graph):
            slot = perm[node]
            local_tuple = gt_hw.global_neuron_to_local(slot)
            core_assignment[node] = local_tuple[0]

        return ClusterAndHwOutput(cluster_assignment=core_assignment.astype(np.int_), num_clusters=num_clusters, hw=input_data.hw_config)

    def get_name(self) -> str:
        return "QAP"