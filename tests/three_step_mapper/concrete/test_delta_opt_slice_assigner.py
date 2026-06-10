import pytest
import numpy as np

from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner


# ==========================================
# MOCK DEPENDENCIES
# ==========================================

class MockHardwareConfig:
    def __init__(self):
        self.total_neurons = 5
        self.max_distance = 2

    def num_slices_at_distance(self, d):
        # Dist 1 has 2 slices, Dist 2 has 1 slice
        if d == 1: return 2
        if d == 2: return 1
        return 0

    def core_distance(self, c1, c2):
        return abs(c1 - c2)

    def get_slice_idx(self, dist, local_id):
        # Simple deterministic slice assignment for testing
        return local_id % self.num_slices_at_distance(dist)


class MockGraph:
    def __init__(self, num_vertices, edges):
        self._num_vertices = num_vertices
        self._in_edges = {i: [] for i in range(num_vertices)}
        self._out_edges = {i: [] for i in range(num_vertices)}
        for src, tgt in edges:
            self._out_edges[src].append(tgt)
            self._in_edges[tgt].append(src)

    def num_vertices(self):
        return self._num_vertices

    def get_in_neighbors(self, v):
        return self._in_edges[v]

    def get_out_neighbors(self, v):
        return self._out_edges[v]


class MockSAState:
    def __init__(self, hw_config, core_assignment, local_assignment):
        self.hw_config = hw_config
        self.core_assignment = core_assignment
        self.local_assignment = local_assignment


# ==========================================
# TEST CLASS
# ==========================================

class TestDeltaOptimalSliceAssigner:

    @pytest.fixture
    def setup_data(self):
        """
        Provides a baseline configuration for tests:
        5 nodes (0 to 4).
        Edges: (0->2), (1->2), (3->4)
        """
        hw_config = MockHardwareConfig()
        graph = MockGraph(num_vertices=5, edges=[(0, 2), (1, 2), (3, 4)])

        # Initial State
        initial_cores = np.array([0, 0, 1, 2, 0], dtype=np.int_)
        initial_locals = np.array([0, 1, 0, 0, 0], dtype=np.int_)

        return hw_config, graph, initial_cores, initial_locals

    def test_initialization(self, setup_data):
        hw_config, graph, cores, locals = setup_data

        assigner = DeltaOptimalSliceAssigner(hw_config, graph, cores, locals)

        # Target node 2 has incoming edges from node 0 and node 1
        # Node 2 core = 1. Node 0 core = 0 (dist 1). Node 1 core = 0 (dist 1).
        # Node 0 local = 0 -> slice idx = 0 % 2 = 0
        # Node 1 local = 1 -> slice idx = 1 % 2 = 1
        # So node 2, at distance 1, should have 1 count for slice 0, and 1 count for slice 1

        assert assigner.connection_counts[2, 1, 0] == 1
        assert assigner.connection_counts[2, 1, 1] == 1

        # Argmax of [1, 1] is 0 (first index)
        assert assigner.slice_assignment[2, 1] == 0

        # Target node 4 has incoming from node 3
        # Node 4 core = 0. Node 3 core = 2 (dist 2).
        # Node 3 local = 0 -> slice idx = 0 % 1 = 0
        assert assigner.connection_counts[4, 2, 0] == 1
        assert assigner.slice_assignment[4, 2] == 0

    def test_delta_assign_slices(self, setup_data):
        hw_config, graph, cores, locals = setup_data

        # Initialize assigner
        assigner = DeltaOptimalSliceAssigner(hw_config, graph, cores, locals)

        # Simulate moving Node 1
        # Old state: Node 1 at core 0, local 1
        moved_nodes = [1]
        old_info = {1: (0, 1)}

        # New state: Node 1 moves to core 2, local 0
        new_cores = np.array([0, 2, 1, 2, 0], dtype=np.int_)
        new_locals = np.array([0, 0, 0, 0, 0], dtype=np.int_)
        new_state = MockSAState(hw_config, new_cores, new_locals)

        # Apply delta update
        assigner.delta_assign_slices(new_state, moved_nodes, graph, old_info)

        # Validate target node 2 update:
        # Node 2 core is still 1.
        # Source 1 moved from core 0 (dist 1, slice 1) to core 2 (dist 1, slice 0).
        # Target 2 should now have 2 counts for slice 0, and 0 counts for slice 1.
        assert assigner.connection_counts[2, 1, 0] == 2
        assert assigner.connection_counts[2, 1, 1] == 0
        assert assigner.slice_assignment[2, 1] == 0

    def test_undo_assign_slices(self, setup_data):
        hw_config, graph, cores, locals = setup_data

        assigner = DeltaOptimalSliceAssigner(hw_config, graph, cores, locals)

        # Save exact copies of original matrices
        orig_counts = np.copy(assigner.connection_counts)
        orig_slices = np.copy(assigner.slice_assignment)

        # Apply a move
        moved_nodes = [1]
        old_info = {1: (0, 1)}
        new_cores = np.array([0, 2, 1, 2, 0], dtype=np.int_)
        new_locals = np.array([0, 0, 0, 0, 0], dtype=np.int_)
        new_state = MockSAState(hw_config, new_cores, new_locals)

        assigner.delta_assign_slices(new_state, moved_nodes, graph, old_info)

        # Undo the move
        assigner.undo_assign_slices()

        # Assert state is perfectly restored
        np.testing.assert_array_equal(assigner.connection_counts, orig_counts)
        np.testing.assert_array_equal(assigner.slice_assignment, orig_slices)

    def test_delta_matches_full_recompute(self, setup_data):
        """
        Robustness check: The result of a delta update should be exactly identical
        to throwing away the object and initializing a new one from scratch.
        """
        hw_config, graph, cores, locals = setup_data

        # 1. Delta Route
        delta_assigner = DeltaOptimalSliceAssigner(hw_config, graph, cores, locals)
        moved_nodes = [1, 3]
        old_info = {1: (0, 1), 3: (2, 0)}

        new_cores = np.array([0, 2, 1, 1, 0], dtype=np.int_)  # 1 to core 2, 3 to core 1
        new_locals = np.array([0, 0, 0, 1, 0], dtype=np.int_)  # 1 to local 0, 3 to local 1
        new_state = MockSAState(hw_config, new_cores, new_locals)

        delta_assigner.delta_assign_slices(new_state, moved_nodes, graph, old_info)

        # 2. Scratch Route
        scratch_assigner = DeltaOptimalSliceAssigner(hw_config, graph, new_cores, new_locals)

        # 3. Compare
        np.testing.assert_array_equal(
            delta_assigner.connection_counts,
            scratch_assigner.connection_counts,
            err_msg="Delta connection counts do not match full recomputation!"
        )
        np.testing.assert_array_equal(
            delta_assigner.slice_assignment,
            scratch_assigner.slice_assignment,
            err_msg="Delta slice assignment does not match full recomputation!"
        )