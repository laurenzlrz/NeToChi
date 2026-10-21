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
        hw = input_data.hw_config
        N_graph = graph.num_vertices()
        N_hw = hw.total_neurons

        # 1. Adjacency (transpose is irrelevant since B is symmetric)
        A = np.zeros((N_hw, N_hw))
        A[:N_graph, :N_graph] = cast(Any, gt.adjacency(graph)).toarray()

        # 2. Affinity, vectorized over cores
        core_of_slot = np.array([hw.global_neuron_to_local(k)[0] for k in range(N_hw)])
        C = hw.total_cores
        core_aff = np.array([[1.0 / hw.num_slices_at_distance(hw.core_distance(a, b))
                              for b in range(C)] for a in range(C)])
        B = core_aff[core_of_slot[:, None], core_of_slot[None, :]]

        # 3. FAQ with random restarts (barycenter init is degenerate for symmetric B)
        rng = np.random.default_rng(input_data.seed)
        best = None
        for _ in range(10):  # e.g. 10
            res = quadratic_assignment(
                A, B, method="faq",
                options={"maximize": True, "P0": "randomized",
                         "rng": rng,  # `random_state` on scipy < 1.15
                         "maxiter": 100, "tol": 1e-4})
            if best is None or res.fun > best.fun:
                best = res

        # 4. Map back
        core_assignment = core_of_slot[best.col_ind[:N_graph]].astype(np.int_)
        return ClusterAndHwOutput(cluster_assignment=core_assignment,
                                  num_clusters=C, hw=hw)

    def get_name(self) -> str:
        return "QAP"