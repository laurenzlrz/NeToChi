from typing import Dict

from netochi.mapping.two_step_mapping.interfaces import SliceAssigner, HierarchicalClusterOutput
import graph_tool as gt

from netochi.mapping.two_step_mapping.slice_assignment.slice_assignment_utils import compute_dists_between_cores, compute_core_sizes, compute_best_slice_assignment


class OptimalSliceAssigner(SliceAssigner):

    def assign_slices(self, clustering: HierarchicalClusterOutput, graph: gt.Graph) -> Dict[int, Dict[int, int]]:
        num_neurons = graph.num_vertices()
        core_distances, max_distance = compute_dists_between_cores(cluster_output=clustering)
        core_sizes = compute_core_sizes(cluster_output=clustering)
        neuron_slice_assignments = compute_best_slice_assignment(num_neurons,
                                                                 core_assignment=clustering.cluster_assignment,
                                                                 max_distance=max_distance, core_sizes=core_sizes,
                                                                 core_distances=core_distances, graph=graph)
        return neuron_slice_assignments



