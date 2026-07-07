from typing import List

import graph_tool.all as gt
import icontract
import numpy as np

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw, ClusterAndHwOutput, ClusterFunction


def infer_core_assignment(hierarchy_mappings) -> np.ndarray:
    """
    given the hierarchy mappings, infers the core assignment that corresponds to the hardware cores that are indexed from left to right: 0, ..., C
    """
    # Build top-down parent-to-child maps: children_map[level][parent_id] = [child_ids...]
    children_map = [{} for _ in range(len(hierarchy_mappings))]
    for level in range(1, len(hierarchy_mappings)):
        for child, parent in hierarchy_mappings[level].items():
            children_map[level].setdefault(parent, []).append(child)

    # Walk the tree using DFS to find the structurally ordered leaves (cores)
    ordered_cores = []
    top_level_parents = sorted(list(set(hierarchy_mappings[-1].values())))
    if len(top_level_parents) > 1:
        raise Exception("top level parents are more than one")

    def dfs(level: int, current_node: int):
        if level == 0:
            ordered_cores.append(current_node)
            return

        # Recurse into children at the level below
        for child in children_map[level].get(current_node, []):
            dfs(level - 1, child)

    # Initiate DFS from the highest level
    top_level_index = len(hierarchy_mappings) - 1
    for root_id in top_level_parents:
        dfs(top_level_index, root_id)

    # `ordered_cores` now contains KaMinPar's arbitrary d=0 cluster IDs in strict topological order.
    # Map them to contiguous sequential indices (0, 1, 2, ..., C-1).
    old_core_to_new_core = {old_id: new_id for new_id, old_id in enumerate(ordered_cores)}

    # Finally, assign each original neuron to its new, hierarchy-aware core ID.
    # We initialize an array of length N (assuming neuron IDs are 0 to N-1)
    core_assignment = np.zeros(len(hierarchy_mappings[0]), dtype=int)
    for neuron_id, old_core_id in hierarchy_mappings[0].items():
        core_assignment[neuron_id] = old_core_to_new_core[old_core_id]
    return core_assignment

def build_new_temp_graph(temp_graph, nr_new_vertices, cluster_assignment):
    """
    constructs new graph:
    - nodes correspond to the inferred clusters
    - the number of edges between nodes is the number of edges between nodes in the corresponding clusters
    """
    # Initialize the new graph
    new_temp_graph = gt.Graph(directed=temp_graph.is_directed())
    new_temp_graph.add_vertex(nr_new_vertices)
    # Add edges between the new clusters
    new_edges = []
    for edge in temp_graph.edges():
        src_cluster = cluster_assignment[int(edge.source())]
        tgt_cluster = cluster_assignment[int(edge.target())]
        if src_cluster != tgt_cluster:
            new_edges.append((src_cluster, tgt_cluster))
    new_temp_graph.add_edge_list(new_edges)
    return new_temp_graph


class BottomUpClusterer(ClustererFixedHw):
    """
    Mapper that uses a bottom-up approach using a given cluster function.
´   """

    @icontract.require(lambda clusterer: isinstance(clusterer, ClusterFunction))
    def __init__(self, clusterer: ClusterFunction) -> None:
        self.clusterer = clusterer

    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        hw = input_data.hw_config
        graph = input_data.graph
        temp_graph = graph
        hierarchy_mappings: List[np.ndarray] = []

        for d in range(hw.router_levels + 1):
            print(f"  [Log] reached distance {d}/{hw.router_levels}")

            # Cluster temp graph
            cluster_size = input_data.hw_config.neurons_per_core if d == 0 else input_data.hw_config.nodes_per_router
            nr_clusters = int(temp_graph.num_vertices() / cluster_size)
            temp_clustering: ClusterAndHwOutput = self.clusterer(graph=temp_graph, num_clusters=nr_clusters)
            hierarchy_mappings.append(temp_clustering.cluster_assignment)

            # build new temp_graph for the next hierarchy level
            temp_graph = build_new_temp_graph(temp_graph=temp_graph, nr_new_vertices=nr_clusters, cluster_assignment=temp_clustering.cluster_assignment)

        # infer core assignment from hierarchical clustering
        core_assignment: np.ndarray = infer_core_assignment(hierarchy_mappings=hierarchy_mappings)

        return ClusterAndHwOutput(cluster_assignment=core_assignment, num_clusters=hw.total_cores, hw=hw)

    def get_name(self) -> str:
        return "BottomUp"

