
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import graph_tool as gt
from collections import defaultdict

from netochi.mapping.two_step_mapping.interfaces import ClusterOutput, HierarchicalClusterOutput


@dataclass
class DistanceMatrix:
    distances: Dict[Tuple[int, int], int] #(core_id_1, core_id_2) -> dist

    def get(self, src: int, dst: int) -> int:
        return self.distances[(src, dst)]


class DistanceCalculator:
    """
    Implements Algorithm 2 from Leite's Thesis.

    dist(i, j) = floor(n / e(i, j)) + 1

    where:
        n       = neurons per core
        e(i,j)  = number of connections core i receives from core j
    """

    def __init__(self, graph: gt.Graph, cluster_output: ClusterOutput):
        self._dist_normalization_const = 20
        self._graph = graph
        self._cluster_output = cluster_output
        self._symmetric_policy = "closest"


    def compute_dists(self) -> DistanceMatrix:
        core_edges = self._count_connections_between_cores()
        num_cores = self._cluster_output.num_clusters

        dist: Dict[Tuple[int, int], int] = {}
        for i in range(num_cores):
            for j in range(num_cores):
                if i == j:
                    dist[(i, j)] = 0
                    continue

                e_ij = core_edges[(i, j)]

                if e_ij <= 0:
                    dist[(i, j)] = self._dist_normalization_const + 1
                    continue

                d = self._dist_normalization_const * math.floor(1 / e_ij) + 1
                dist[(i, j)] = d

        self._symmetrize(dist, num_cores)

        return DistanceMatrix(distances=dist)


    def _count_connections_between_cores(self):
        """
        Returns:
            edge_count[(dst_core, src_core)] = number of connections
        """
        edge_count = defaultdict(int)
        for edge in self._graph.edges():
            src = int(edge.source())
            dst = int(edge.target())
            src_core = self._cluster_output.cluster_assignment[src]
            dst_core = self._cluster_output.cluster_assignment[dst]
            if src_core == dst_core:
                continue
            edge_count[(dst_core, src_core)] += 1
        return edge_count


    def _symmetrize(self, dist, num_cores):
        for i in range(num_cores):
            for j in range(i + 1, num_cores):
                d_ij = dist[(i, j)]
                d_ji = dist[(j, i)]

                if d_ij == d_ji:
                    continue

                if self._symmetric_policy == "closest":
                    d = min(d_ij, d_ji)
                else:
                    d = max(d_ij, d_ji)

                dist[(i, j)] = d
                dist[(j, i)] = d

def compute_best_slice_assignment(num_neurons: int, core_assignment, max_distance, core_sizes, core_distances, graph: gt.Graph):
    """
    core_assignment: neuron_id -> core_id
    """

    edge_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # edge_counter[target][distance][slice_id] = count

    for edge in graph.edges():
        u = int(edge.source())
        v = int(edge.target())
        C_u = core_assignment[u]
        C_v = core_assignment[v]
        core_dist = core_distances[C_u][C_v]
        slice_id = get_slice_id(C_u, u, core_dist, core_sizes)
        edge_counter[v][core_dist][slice_id] += 1

    slice_assignment = defaultdict(dict)  # slice_assignment[target][distance] = best_slice_id
    for v in range(num_neurons):
        for d in range(max_distance):
            if d not in edge_counter[v]:
                slice_assignment[v][d] = 0
                continue
            best_slice = max(
                edge_counter[v][d],
                key=edge_counter[v][d].get
            )
            slice_assignment[v][d] = best_slice

    return slice_assignment

def get_slice_id(core_u, u, core_dist, core_sizes, local_assignment):
    num_slices = 2 ** core_dist
    return (local_assignment[u] * num_slices) // core_sizes[core_u]



def compute_dists_between_cores(cluster_output: HierarchicalClusterOutput) -> Tuple[List[List[int]], int]:
    """
    Compute pairwise distances between cores.

    Distance definition:
        Number of hops upward to the lowest common ancestor (LCA).

    Assumptions:
    - All leaves are at the same depth.
    - cluster_assignment maps:
          core_id -> leaf_cluster_id
    - cluster_parent maps:
          cluster_id -> parent_cluster_id
      with root parent == -1.
    """
    cluster_parent = cluster_output.cluster_parent
    cluster_assignment = cluster_output.cluster_assignment
    num_cores = cluster_output.num_clusters
    dists = [[0 for _ in range(num_cores)] for _ in range(num_cores)]
    max_dist = 0
    for i in range(num_cores):
        for j in range(i + 1, num_cores):
            c_i = cluster_assignment[i]
            c_j = cluster_assignment[j]
            dist = 0
            # Walk upward until common ancestor found
            while c_i != c_j:
                dist += 1
                c_i = cluster_parent.get(c_i, -1)
                c_j = cluster_parent.get(c_j, -1)
            dists[i][j] = dist
            dists[j][i] = dist
            if dist > max_dist:
                max_dist = dist
    return dists, max_dist



def compute_core_sizes(cluster_output: HierarchicalClusterOutput) -> Dict[int, int]:
        """
        Compute the size of each cluster.
        Returns: cluster_id -> number of assigned nodes
        """
        cluster_sizes: Dict[int, int] = defaultdict(int)
        for _, cluster_id in cluster_output.cluster_assignment.items():
            cluster_sizes[cluster_id] += 1
        return dict(cluster_sizes)