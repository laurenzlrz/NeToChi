
from typing import Optional
import graph_tool.all as gt

from netochi.mapping.simulated_annealing_fix_hw.sa_state import SAState
from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner


class Mutation:

    def do(self, state: SAState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph):
        raise NotImplementedError()

    def undo(self, state: SAState, slice_assigner: DeltaOptimalSliceAssigner):
        raise NotImplementedError()


class MoveMutation(Mutation):

    def __init__(self, node: int, new_core: int, new_local: int):
        self.node: int = node
        self.new_core: int = new_core
        self.new_local_address: int = new_local
        self.old_core: int = -1
        self.old_local_address: int = -1

    def do(self, state: SAState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph):
        # 1. update undo information
        self.old_core = int(state.core_assignment[self.node])
        self.old_local_address = int(state.local_assignment[self.node])
        # 2. apply move
        self._apply_move(state, new_core=self.new_core, new_local_address=self.new_local_address)
        # 3. update slice assigner
        slice_assigner.delta_assign_slices(state=state, moved_nodes=[self.node], graph=graph, old_core_and_local_of_moved_nodes={int(self.node): (self.old_core, self.old_local_address)})

    def undo(self, state: SAState, slice_assigner: DeltaOptimalSliceAssigner):
        self._apply_move(state, new_core=self.old_core, new_local_address=self.old_local_address)
        slice_assigner.undo_assign_slices()

    def _apply_move(self, state: SAState, new_core, new_local_address):
        old_core = state.core_assignment[self.node]
        old_local = state.local_assignment[self.node]
        state.core_assignment[self.node] = new_core
        state.local_assignment[self.node] = new_local_address
        state.slot_to_node[old_core, old_local] = -1
        state.slot_to_node[new_core, new_local_address] = self.node


class SwapMutation(Mutation):

    def __init__(self, node_a, node_b):
        self.node_a = node_a
        self.node_b = node_b

    def do(self, state: SAState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph):
        old_core_a, old_local_a = int(state.core_assignment[self.node_a]), int(state.local_assignment[self.node_a])
        old_core_b, old_local_b = int(state.core_assignment[self.node_b]), int(state.local_assignment[self.node_b])
        self._apply_swap(state)
        slice_assigner.delta_assign_slices(state=state, moved_nodes=[self.node_a, self.node_b], graph=graph,
                                           old_core_and_local_of_moved_nodes={int(self.node_a): (old_core_a, old_local_a), int(self.node_b): (old_core_b, old_local_b)})

    def undo(self, state: SAState, slice_assigner: DeltaOptimalSliceAssigner):
        self._apply_swap(state)
        slice_assigner.undo_assign_slices()

    def _apply_swap(self, state: SAState):
        core_a, local_a = state.core_assignment[self.node_a], state.local_assignment[self.node_a]
        core_b, local_b = state.core_assignment[self.node_b], state.local_assignment[self.node_b]
        state.core_assignment[self.node_a] = core_b
        state.core_assignment[self.node_b] = core_a
        state.local_assignment[self.node_a] = local_b
        state.local_assignment[self.node_b] = local_a
        state.slot_to_node[core_a, local_a] = self.node_b
        state.slot_to_node[core_b, local_b] = self.node_a







