import os
from typing import List

import graph_tool as gt
import kahypar

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.clustering.clusterer.kaHyPar_cluster import KaHyParHierarchicalClusterer


class KaHyParHyperedgesClusterer(KaHyParHierarchicalClusterer):
    """
    uses hyper-edges
    """

    def _bisect_cluster(self, graph: gt.Graph, nodes_in_cluster: List[int], mapping_input: MosaicMappingInput) -> List[List[int]]:
        """
        Extracts an induced cluster subgraph, converts incoming edges per node
        into single hyperedges, and computes a k-way partition via KaHyPar.
        """
        ordered_nodes = list(set(nodes_in_cluster))

        # KaHyPar requires 0-based contiguous indices for nodes
        local_map = {v_id: i for i, v_id in enumerate(ordered_nodes)}

        hyperedge_indices = [0]
        hyperedges = []
        num_nets = 0

        # FIX: Check if the graph is directed or undirected
        is_directed = graph.is_directed()

        # Build hyperedges: 1 net per node, containing neighbors + the node itself
        for v_id in ordered_nodes:
            v_obj = graph.vertex(v_id)

            # FIX: Use all_neighbors() if undirected, as in_neighbors() is empty in graph-tool for undirected graphs
            neighbors_iter = v_obj.in_neighbors() if is_directed else v_obj.all_neighbors()
            in_neighbors = [int(n) for n in neighbors_iter if int(n) in local_map]

            if in_neighbors:
                # Group the target node and its sources into a single hyperedge
                net_pins = set(in_neighbors)
                net_pins.add(v_id)

                # Only register the net if it connects at least two distinct nodes
                if len(net_pins) > 1:
                    for pin in net_pins:
                        hyperedges.append(local_map[pin])

                    num_nets += 1
                    hyperedge_indices.append(len(hyperedges))

        num_vertices = len(ordered_nodes)
        k = mapping_input.hw_config.nodes_per_router

        # Robustness fallback: Even splitting
        def build_fallback():
            fallback_partitions = [[] for _ in range(k)]
            for i, node in enumerate(ordered_nodes):
                fallback_partitions[i % k].append(node)
            return fallback_partitions

        if num_nets == 0:
            return build_fallback()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "cut_kKaHyPar_sea20.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"KaHyPar config file not found at: {config_path}. "
                f"Please provide an absolute path or ensure the file exists in the working directory."
            )

        try:
            # Initialize KaHyPar hypergraph
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

            # Route native results back to global graph scope IDs
            partitions = [[] for _ in range(k)]
            for local_idx in range(num_vertices):
                block_id = hypergraph.blockID(local_idx)
                # Safety fallback in case KaHyPar returns an out-of-bounds block ID
                if 0 <= block_id < k:
                    partitions[block_id].append(ordered_nodes[local_idx])
                else:
                    partitions[0].append(ordered_nodes[local_idx])

            return partitions

        except Exception as e:
            print(f"KaHyPar partitioning failed: {e}. Falling back to naive k-split.")
            return build_fallback()

    def get_name(self):
        return "KaHyParHyperedges"