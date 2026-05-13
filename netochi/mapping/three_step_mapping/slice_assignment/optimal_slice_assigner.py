from typing import Dict

from mypy.checkexpr import defaultdict

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.interfaces import SliceAssigner, HierarchicalClusterOutput, \
    HierarchicalClusterOutputAndHw
import graph_tool as gt

from netochi.mapping.three_step_mapping.slice_assignment.slice_assignment_utils import compute_dists_between_cores, compute_core_sizes, compute_best_slice_assignment


class OptimalSliceAssigner(SliceAssigner):

    def assign_slices(self, clustering: HierarchicalClusterOutputAndHw, graph: gt.Graph, local_assignment: Dict[int, int], hw: MosaicHardwareConfig) -> Dict[int, Dict[int, int]]:
        s_assignment = defaultdict(dict)
        for tgt in range(graph.num_vertices()):
            tgt_core = clustering.cluster_assignment[tgt]
            for d in range(1, hw.max_distance + 1):
                best_slice = 0
                max_sources = -1
                n_slices = hw.num_slices_at_distance(d)

                for s_idx in range(n_slices):
                    count = 0
                    start, end = hw.get_slice_bounds(d, s_idx)
                    for src in graph.get_in_neighbors(tgt):
                        src_core = clustering.cluster_assignment[src]
                        if hw.core_distance(tgt_core, src_core) == d:
                            if start <= local_assignment[src] < end:
                                count += 1
                    if count > max_sources:
                        max_sources = count
                        best_slice = s_idx
                s_assignment[tgt][d] = best_slice
        return s_assignment



