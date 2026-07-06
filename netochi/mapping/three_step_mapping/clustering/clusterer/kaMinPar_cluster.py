import os
import tempfile
import collections
from typing import List
import numpy as np
import graph_tool.all as gt

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw, ClusterAndHwOutput

import kaminpar

import numpy as np
from scipy.sparse import csr_matrix
import numpy.typing as npt


def compute_normalized_cut(adj_matrix: csr_matrix, labels: npt.NDArray[np.int_]) -> float:
    """
    Computes the Normalized Cut for a given graph and cluster assignment.

    :param adj_matrix: A SciPy sparse adjacency matrix of the graph.
    :param labels: A NumPy array of cluster labels for each node.
    :return: The total Normalized Cut score (lower is better).
    """
    unique_labels = np.unique(labels)
    total_ncut = 0.0

    # Precompute the degree of each node (sum across rows)
    # .A1 flattens the matrix into a 1D NumPy array
    degrees = adj_matrix.sum(axis=1).A1

    for k in unique_labels:
        # Boolean masks for nodes inside and outside the current cluster
        in_cluster = (labels == k)
        out_cluster = (labels != k)

        # Volume: sum of degrees of all nodes in this cluster
        vol = degrees[in_cluster].sum()

        # Guard against zero-volume clusters (isolated nodes with no edges)
        if vol == 0:
            continue

        # Cut: sum of edge weights between nodes in the cluster and outside the cluster
        # We index the sparse matrix directly for efficiency
        cut = adj_matrix[in_cluster, :][:, out_cluster].sum()

        # Accumulate the ratio
        total_ncut += cut / vol

    return total_ncut

class KaMinParHierarchicalClusterer(ClustererFixedHw):
    """
    An implementation of HierarchicalClusterer that performs top-down
    hierarchical graph clustering using the parallel KaMinPar algorithm.
    """

    def __init__(self, imbalance: float = 0.001, num_threads: int = 1) -> None:
        """
        Args:
            imbalance: The maximum allowed load imbalance factor for KaMinPar (e.g., 0.03 = 3%).
            num_threads: Number of parallel threads for KaMinPar to use.
        """
        self.imbalance = imbalance
        self.num_threads = num_threads

    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        graph = input_data.graph
        num_nodes = graph.num_vertices()

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

            # Partition the current cluster using KaMinPar
            partitions = self._partition_cluster(graph, nodes_in_cluster=nodes_in_cluster, nr_clusters=input_data.hw_config.nodes_per_router)

            # Ensure the split was successful and non-trivial
            populated_partitions = sum(1 for p in partitions if len(p) > 0)
            if populated_partitions != input_data.hw_config.nodes_per_router:
                raise RuntimeError("kyHyPar did not return exactly k partitions")

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


        if input_data.assignment is not None:
            adj_matrix = gt.adjacency(input_data.graph)
            gt_labels = np.array(input_data.assignment.neuron_core_pre_assignment, dtype=np.int_)
            algo_labels = cluster_assignment

            ncut_gt = compute_normalized_cut(adj_matrix, gt_labels)
            ncut_algo = compute_normalized_cut(adj_matrix, algo_labels)

            print(f"  --> kaMinPar: ground truth: {ncut_gt}, algo: {ncut_algo} (normalized cut, to minimize)")


        return ClusterAndHwOutput(
            cluster_assignment=cluster_assignment,
            num_clusters=len(ordered_leaves),
            hw=input_data.hw_config
        )

    def _partition_cluster(self, graph: gt.Graph, nr_clusters: int, nodes_in_cluster: List[int] | None) -> List[List[int]]:
        """
        Extracts an induced cluster subgraph, encodes it to METIS format,
        and calls KaMinPar to compute a k-way partition.
        """
        if nodes_in_cluster is None:
            # graph.get_vertices() returns a numpy array of all integer vertex IDs
            nodes_in_cluster = graph.get_vertices().tolist()
        ordered_nodes = list(set(nodes_in_cluster))
        local_map = {v_id: i + 1 for i, v_id in enumerate(ordered_nodes)}  # METIS requires 1-based indexing

        metis_lines = []
        total_edges = 0

        for v_id in ordered_nodes:
            v_obj = graph.vertex(v_id)

            # 1. Collect ALL neighbors (in and out), keeping duplicates
            all_neighbors = [int(n) for n in v_obj.out_neighbors()] + [int(n) for n in v_obj.in_neighbors()]

            # 2. Filter for cluster membership and discard self-loops
            valid_neighbors = [local_map[n] for n in all_neighbors if n in local_map and n != v_id]

            # 3. Count frequencies (multi-edges become weights)
            neighbor_weights = collections.Counter(valid_neighbors)

            total_edges += len(neighbor_weights)

            # 4. Format for weighted METIS: neighbor1 weight1 neighbor2 weight2 ...
            line_parts = []
            for neighbor_id, weight in neighbor_weights.items():
                line_parts.extend([neighbor_id, weight])

            metis_lines.append(" ".join(map(str, line_parts)))

        num_vertices = len(ordered_nodes)
        num_edges = total_edges // 2  # Undirected tracking fallback

        # Robustness fallback function for trivial splits
        def build_fallback():
            fallback = [[] for _ in range(nr_clusters)]
            for i, node in enumerate(ordered_nodes):
                fallback[i % nr_clusters].append(node)
            return fallback

        # If sub-graph has no edges, split as evenly as possible into k chunks
        if num_edges == 0:
            return build_fallback()

        # Write out to a transient file descriptor safely managed by the OS
        with tempfile.NamedTemporaryFile(mode="w", suffix=".metis", delete=False) as tmp:
            tmp.write(f"{num_vertices} {num_edges} 001\n") # 001 tells the METIS parser: "This graph has edge weights"
            tmp.write("\n".join(metis_lines) + "\n")
            tmp_path = tmp.name

        try:
            # Configure and instantiate the wrapper runtime
            ctx = kaminpar.default_context()
            # Note: Depending on your KaMinPar Python bindings, you might be able
            # to pass ctx.quiet = True or similar here to suppress C++ output!
            instance = kaminpar.KaMinPar(num_threads=self.num_threads, ctx=ctx)

            # Load METIS stream data cleanly into native C++ structures
            kp_graph = kaminpar.load_graph(tmp_path, kaminpar.GraphFileFormat.METIS, compress=False)

            # Partition with k blocks
            partition = instance.compute_partition(kp_graph, k=nr_clusters, eps=self.imbalance)

            # Sort native results back into global graph scope dynamically
            partitions = [[] for _ in range(nr_clusters)]
            for local_idx, block_id in enumerate(partition):
                if 0 <= block_id < nr_clusters:
                    partitions[block_id].append(ordered_nodes[local_idx])
                else:
                    # Safety net in case KaMinPar yields an out-of-bounds block ID
                    partitions[0].append(ordered_nodes[local_idx])

            return partitions

        except Exception as e:
            print(f"KaMinPar partitioning failed: {e}. Falling back to naive k-split.")
            return build_fallback()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_name(self) -> str:
        return "KaMinPar"