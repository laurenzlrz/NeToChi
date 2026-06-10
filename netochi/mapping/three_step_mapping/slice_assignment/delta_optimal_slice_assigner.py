from typing import List, Any

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.simulated_annealing_fix_hw.sa_state import SAState
import graph_tool as gt
import numpy as np



class DeltaOptimalSliceAssigner:
    """
    infers optimal slice assignment given:
     - clustering
     - hw
     - local address assignment
     - previous opt. assignment + delta (swapped nodes)
    """

    def __init__(self, hw_config: MosaicHardwareConfig, graph: gt.Graph, cluster_assignment: np.ndarray[int], local_assignment: np.ndarray[int]):
        num_nodes = hw_config.total_neurons
        max_dist = hw_config.max_distance
        max_slices = max(hw_config.num_slices_at_distance(d) for d in range(1, max_dist + 1))

        # these data structures will be cached and reused for the delta updates
        self.slice_assignment = np.zeros((num_nodes, max_dist + 1), dtype=np.int_)
        self.connection_counts = np.zeros((num_nodes, max_dist + 1, max_slices), dtype=np.int_)

        # 1. compute the connection_counts matrix
        for tgt in range(graph.num_vertices()):
            tgt_core = cluster_assignment[tgt]

            for src in graph.get_in_neighbors(tgt):
                src_core = cluster_assignment[src]
                dist = hw_config.core_distance(src_core, tgt_core)
                if dist > 0: # We only track slices for dist > 0, because for dist=0, the slice is fix
                    s_idx = hw_config.get_slice_idx(dist, local_assignment[src])
                    self.connection_counts[tgt, dist, s_idx] += 1

        # 2. Compute the optimal slice assignment using argmax
        for tgt in range(graph.num_vertices()):
            for d in range(1, max_dist + 1):
                n_slices = hw_config.num_slices_at_distance(d)
                if n_slices > 0:
                    # Slice the array [:n_slices] to ignore padding zeros for shorter distances
                    best_slice = np.argmax(self.connection_counts[tgt, d, :n_slices])
                    self.slice_assignment[tgt, d] = best_slice

        # 3. create backup
        self.backup_slice_assignment = None
        self.backup_connection_counts = None

    def delta_assign_slices(self, state: SAState, moved_nodes: List[int],  graph: gt.Graph, old_core_and_local_of_moved_nodes: dict[int, tuple[int, int]]) -> np.ndarray[tuple[Any, Any], np.dtype[np.int_]]:
        """
        for every target neuron and every distance: goes through every possible slice and counts
        """
        # --- 1. create backup ---
        self.backup_slice_assignment = {}
        self.backup_connection_counts = {}

        # Gather all affected nodes to back up
        affected_nodes = set(moved_nodes)
        for src in moved_nodes:
            affected_nodes.update(graph.get_out_neighbors(src))

        # Only copy the specific rows that will change
        for node in affected_nodes:
            self.backup_slice_assignment[node] = np.copy(self.slice_assignment[node])
            self.backup_connection_counts[node] = np.copy(self.connection_counts[node])

        # --- 2. update moved nodes ---
        for tgt in moved_nodes:
            self.connection_counts[tgt].fill(0)  # Clear out completely: fully recompute
            tgt_core = state.core_assignment[tgt]

            for src in graph.get_in_neighbors(tgt):
                src_core = state.core_assignment[src]
                dist = state.hw_config.core_distance(src_core, tgt_core)
                if dist > 0:  # We only track slices for dist > 0, because for dist=0, the slice is fix
                    s_idx = state.hw_config.get_slice_idx(dist, state.local_assignment[src])
                    self.connection_counts[tgt, dist, s_idx] += 1

            # Resolve maximizing slices
            for d in range(1, state.hw_config.max_distance + 1):
                n_slices = state.hw_config.num_slices_at_distance(d)
                if n_slices > 0:
                    # Slice the array [:n_slices] to ignore padding zeros for shorter distances
                    best_slice = np.argmax(self.connection_counts[tgt, d, :n_slices])
                    self.slice_assignment[tgt, d] = best_slice

        # --- 3. update target nodes of moved nodes ---
        for src in moved_nodes:
            for tgt in graph.get_out_neighbors(src):

                if tgt in moved_nodes:
                    continue  # Skip; handled entirely in Phase 2

                old_core_src, old_local_src = old_core_and_local_of_moved_nodes[src]

                old_dist = state.hw_config.core_distance(old_core_src, state.core_assignment[tgt])
                new_dist = state.hw_config.core_distance(state.core_assignment[src], state.core_assignment[tgt])

                if old_dist > 0:
                    old_slice_src = state.hw_config.get_slice_idx(old_dist, old_local_src)
                    self.connection_counts[tgt][old_dist][old_slice_src] -= 1
                    # update slice assignment for old distance
                    n_slices_old = state.hw_config.num_slices_at_distance(old_dist)
                    self.slice_assignment[tgt, old_dist] = np.argmax(self.connection_counts[tgt, old_dist, :n_slices_old])

                if new_dist > 0:
                    new_slice_src = state.hw_config.get_slice_idx(new_dist, state.local_assignment[src])
                    self.connection_counts[tgt][new_dist][new_slice_src] += 1
                    # update slice assignment for new distance
                    n_slices_new = state.hw_config.num_slices_at_distance(new_dist)
                    self.slice_assignment[tgt, new_dist] = np.argmax(self.connection_counts[tgt, new_dist, :n_slices_new])

        return self.slice_assignment

    def undo_assign_slices(self):
        for node, row in self.backup_slice_assignment.items():
            self.slice_assignment[node] = row
            self.connection_counts[node] = self.backup_connection_counts[node]


