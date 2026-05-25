
import numpy as np
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig



class SAState:

    def __init__(self, num_nodes: int, hw_config: MosaicHardwareConfig):
        self.hw_config = hw_config

        num_slots = hw_config.total_cores * hw_config.neurons_per_core
        if num_slots < num_nodes:
            raise ValueError("Not enough hardware slots for the input graph nodes.")

        # random initial assignment
        initial_flat_slots = np.random.choice(num_slots, size=num_nodes, replace=False)
        self.core_assignment = initial_flat_slots // hw_config.neurons_per_core
        self.local_assignment = initial_flat_slots % hw_config.neurons_per_core

        self.slot_to_node = np.full((hw_config.total_cores, hw_config.neurons_per_core), -1, dtype=np.int_)
        self.slot_to_node[self.core_assignment, self.local_assignment] = np.arange(num_nodes)



class Mutation:

    def do(self, state: SAState):
        raise NotImplementedError()

    def undo(self, state: SAState):
        raise NotImplementedError()


class MoveMutation(Mutation):

    def __init__(self, node, new_core, new_local):
        self.node = node
        self.new_core = new_core
        self.new_local_address = new_local
        self.old_core = None
        self.old_local_address = None

    def do(self, state: SAState):
        # 1. update undo information
        self.old_core = state.core_assignment[self.node]
        self.old_local_address = state.local_assignment[self.node]
        # 2. apply move
        self._apply_move(state, new_core=self.new_core, new_local_address=self.new_local_address)

    def undo(self, state: SAState):
        self._apply_move(state, new_core=self.old_core, new_local_address=self.old_local_address)

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

    def do(self, state: SAState):
        self._apply_swap(state)

    def undo(self, state: SAState):
        self._apply_swap(state)

    def _apply_swap(self, state: SAState):
        core_a, local_a = state.core_assignment[self.node_a], state.local_assignment[self.node_a]
        core_b, local_b = state.core_assignment[self.node_b], state.local_assignment[self.node_b]
        state.core_assignment[self.node_a] = core_b
        state.core_assignment[self.node_b] = core_a
        state.local_assignment[self.node_a] = local_b
        state.local_assignment[self.node_b] = local_a
        state.slot_to_node[core_a, local_a] = self.node_b
        state.slot_to_node[core_b, local_b] = self.node_a






