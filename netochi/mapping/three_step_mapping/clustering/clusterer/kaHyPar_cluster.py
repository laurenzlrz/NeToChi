import collections
import os
from typing import List
import numpy as np
import graph_tool as gt

from netochi.input_generator.interfaces import MosaicMappingInput, HWMappingInput
from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw, ClusterAndHwOutput

import kahypar


class KaHyParHierarchicalClusterer(ClustererFixedHw):
    """
    An implementation of HierarchicalClusterer that performs top-down
    hierarchical graph clustering using the KaHyPar hypergraph partitioning algorithm.
    """

    def __init__(self, imbalance: float = 0.03) -> None:
        """
        Args:
            imbalance: The maximum allowed load imbalance factor (e.g., 0.03 = 3%).
            config_path: Path to a valid KaHyPar INI configuration file.
                         (Required by KaHyPar's context initialization).
        """
        self.imbalance = imbalance

    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        graph = input_data.graph
        num_nodes = graph.num_vertices()

        # Handle empty or trivial graph edges gracefully
        if num_nodes == 0:
            return ClusterAndHwOutput(
                cluster_assignment=np.array([], dtype=np.int_),
                num_clusters=1,
                hw=input_data.hw_config
            )

        # Structure to hold our hierarchy tree
        # Root cluster is index 0, parent is -1
        cluster_parent = [-1]
        cluster_nodes = {0: list(range(num_nodes))}
        is_leaf = {0: True}

        # Queue to handle top-down cluster partitions
        queue = collections.deque([0])
        current_leaves = 1

        while queue and current_leaves < input_data.hw_config.total_cores:
            c = queue.popleft()
            nodes_in_cluster = cluster_nodes[c]

            # Don't attempt to split singletons
            if len(nodes_in_cluster) <= 1:
                continue

            # Partition the current cluster using KaHyPar (returns a list of k lists)
            partitions = self._bisect_cluster(graph, nodes_in_cluster, input_data)

            # Ensure the split was successful and non-trivial
            # (at least 2 child partitions must actually contain nodes)
            populated_partitions = sum(1 for p in partitions if len(p) > 0)
            if populated_partitions < 2:
                continue

            # Mark current node as no longer a leaf
            is_leaf[c] = False

            # Assign nodes to child layers dynamically
            for p_nodes in partitions:
                child_id = len(cluster_parent)

                # Register parent
                cluster_parent.append(c)

                # Assign nodes and leaf status
                cluster_nodes[child_id] = p_nodes
                is_leaf[child_id] = True

                # Queue child for further possible splits
                queue.append(child_id)

            # Update leaf count: removed 1 leaf (parent), added k leaves (children)
            current_leaves += (len(partitions) - 1)

        # ====== Map to ClusterAndHwOutput
        # 1. Reconstruct the tree's children for top-down traversal
        children = collections.defaultdict(list)
        for child_id, parent_id in enumerate(cluster_parent):
            if parent_id != -1:
                children[parent_id].append(child_id)

        # 2. Extract leaves in left-to-right (DFS) order
        ordered_leaves = []

        def dfs(node_id: int):
            if is_leaf[node_id]:
                ordered_leaves.append(node_id)
            else:
                for child_id in children[node_id]:
                    dfs(child_id)

        dfs(0)  # Start traversal from root

        # 3. Create a mapping from internal tree ID to flat hardware core ID (0, 1, 2...)
        leaf_to_hw_id = {tree_id: hw_id for hw_id, tree_id in enumerate(ordered_leaves)}

        # 4. Compile final flat cluster assignment map targeting the re-indexed leaves
        cluster_assignment = np.zeros(num_nodes, dtype=np.int_)
        for old_leaf_id, hw_id in leaf_to_hw_id.items():
            for node in cluster_nodes[old_leaf_id]:
                cluster_assignment[node] = hw_id

        return ClusterAndHwOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=len(ordered_leaves),
            hw=input_data.hw_config
        )

    def _bisect_cluster(self, graph: gt.Graph, nodes_in_cluster: List[int], mapping_input: MosaicMappingInput) -> List[
        List[int]]:
        """
        Extracts an induced cluster subgraph, maps it to KaHyPar's in-memory
        hypergraph structures, and computes a k-way partition.
        """
        ordered_nodes = list(set(nodes_in_cluster))
        # KaHyPar requires 0-based contiguous indices for nodes
        local_map = {v_id: i for i, v_id in enumerate(ordered_nodes)}

        hyperedge_indices = [0]
        hyperedges = []
        seen_edges = set()

        # Build local hyperedges (standard graph edges with 2 pins)
        for v_id in ordered_nodes:
            v_obj = graph.vertex(v_id)
            for n_obj in v_obj.out_neighbors():
                n_id = int(n_obj)

                # Only include edges fully contained in this cluster
                if n_id in local_map:
                    # Canonicalize edge to avoid duplicate bidirectional entries
                    edge_tuple = tuple(sorted((v_id, n_id)))

                    if edge_tuple not in seen_edges:
                        seen_edges.add(edge_tuple)
                        hyperedges.append(local_map[v_id])
                        hyperedges.append(local_map[n_id])
                        hyperedge_indices.append(len(hyperedges))

        num_vertices = len(ordered_nodes)
        num_nets = len(seen_edges)

        # Target k partitions
        k = mapping_input.hw_config.nodes_per_router

        # Robustness fallback: If sub-graph has no edges, split as evenly as possible into k chunks
        if num_nets == 0:
            fallback_partitions = [[] for _ in range(k)]
            for i, node in enumerate(ordered_nodes):
                fallback_partitions[i % k].append(node)
            return fallback_partitions

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "cut_kKaHyPar_sea20.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"KaHyPar config file not found at: {config_path}. "
                f"Please provide an absolute path or ensure the file exists in the working directory."
            )

        try:
            # Initialize KaHyPar hypergraph in memory
            # edge weights = 1, node weights = 1
            hypergraph = kahypar.Hypergraph(
                num_vertices,
                num_nets,
                hyperedge_indices,
                hyperedges,
                k,
                [1] * num_nets,
                [1] * num_vertices
            )

            # Setup Context
            context = kahypar.Context()
            context.loadINIconfiguration(config_path)
            context.setK(k)
            context.setEpsilon(self.imbalance)
            context.suppressOutput(True)

            # Perform Partitioning
            kahypar.partition(hypergraph, context)

            # Route native results back to global graph scope IDs dynamically
            partitions = [[] for _ in range(k)]

            for local_idx in range(num_vertices):
                block_id = hypergraph.blockID(local_idx)
                partitions[block_id].append(ordered_nodes[local_idx])

            return partitions

        except Exception as e:
            # Catch unexpected KaHyPar errors and use the fallback
            print(f"KaHyPar partitioning failed: {e}. Falling back to naive k-split.")
            fallback_partitions = [[] for _ in range(k)]
            for i, node in enumerate(ordered_nodes):
                fallback_partitions[i % k].append(node)
            return fallback_partitions

        except Exception as e:
            # Fallback execution in case of config load failures or partition errors
            print(f"KaHyPar bisection failed: {e}. Falling back to mid-split.")
            mid = num_vertices // 2
            return ordered_nodes[:mid], ordered_nodes[mid:]