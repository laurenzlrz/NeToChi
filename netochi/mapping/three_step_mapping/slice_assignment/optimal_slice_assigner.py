from typing import Dict, List

from netochi.mapping.three_step_mapping.interfaces import SliceAssigner, ClusterAndHwOutput
import graph_tool as gt



class OptimalSliceAssigner(SliceAssigner):
    """
    infers optimal slice assignment given clustering, hw, local address assignment
    """

    def assign_slices(self, clustering: ClusterAndHwOutput, graph: gt.Graph, local_assignment: Dict[int, int]) -> List[List[int]]:
        """
        for every target neuron and every distance: goes through every possible slice and counts
        """
        num_targets = graph.num_vertices()
        max_dist = clustering.hw.max_distance

        # Dimensions: [num_targets][max_dist + 1]
        s_assignment = [[0] * (max_dist + 1) for _ in range(num_targets)]

        for tgt in range(graph.num_vertices()):
            tgt_core = clustering.cluster_assignment[tgt]
            for d in range(1, clustering.hw.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = clustering.hw.num_slices_at_distance(d)

                for s_idx in range(n_slices):
                    count = 0
                    start, end = clustering.hw.get_slice_bounds(d, s_idx)
                    for src in graph.get_in_neighbors(tgt):
                        # for every source neuron: checks whether core distance = d and whether its local assignment is in current slice
                        src_core = clustering.cluster_assignment[src]
                        if clustering.hw.core_distance(tgt_core, src_core) == d:
                            if start <= local_assignment[src] < end:
                                count += 1
                    if count > max_sources:
                        max_sources = count
                        best_slice = s_idx
                s_assignment[tgt][d] = best_slice
        return s_assignment



