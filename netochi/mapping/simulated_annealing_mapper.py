import math
import random
from typing import Any

from pydantic import BaseModel

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicHWMappingInput

import numpy as np
import graph_tool.all as gt

from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner

SWAP = 0
MOVE = 1

class SAState:

    def __init__(self, num_nodes: int, hw_config: MosaicHardwareConfig):
        self.hw_config = hw_config

        num_slots = hw_config.total_cores * hw_config.neurons_per_core
        if num_slots < num_nodes:
            raise ValueError("Not enough hardware slots for the input graph nodes.")

        initial_flat_slots = np.random.choice(num_slots, size=num_nodes, replace=False)

        self.core_assignment = initial_flat_slots // hw_config.neurons_per_core
        self.local_assignment = initial_flat_slots % hw_config.neurons_per_core

        self.slot_to_node = np.full((hw_config.total_cores, hw_config.neurons_per_core), -1, dtype=np.int_)
        self.slot_to_node[self.core_assignment, self.local_assignment] = np.arange(num_nodes)



class SimAnnealingMapper(BaseModel, BaseMapper[MosaicNetworkMappingState[Any], MosaicHWMappingInput[Any]]):
    """
    Mapper using simulated annealing. Takes hw config as input.

    State: (injective) mapping neuron_id -> (core_id, local_address)
    Mutation: Swap two neurons or move neuron to empty slot
    Loss: Given a mapping, we can compute the optimal slice assignment efficiently. We can then compute the number of inconsistencies.
    """

    def run(self, mapping_input: MosaicHWMappingInput[Any]) -> MosaicNetworkMappingState[Any]:
        best_state, best_energy = self._simulated_annealing(mapping_input.graph, mapping_input.hw_config)
        clustering: ClusterAndHwOutput = ClusterAndHwOutput(hw=best_state.hw_config, num_clusters=best_state.hw_config.total_cores, cluster_assignment=best_state.core_assignment)
        slice_assignment = OptimalSliceAssigner().assign_slices(clustering=clustering, graph=mapping_input.graph, local_assignment=best_state.local_assignment)
        return MosaicNetworkMappingState(mapping_input=mapping_input, neuron_local_idxs_assignment=best_state.local_assignment, neuron_core_idxs_assignment=best_state.core_assignment, neuron_slice_assignments=slice_assignment)


    def _simulated_annealing(self, graph, hw_config, T_start=10.0, T_min=0.1, alpha=0.95, steps_per_T=50): # gemini: T_min = 10000, alpha=0.98
        state = SAState(graph.num_vertices(), hw_config)
        current_energy = self._compute_energy(state=state, graph=graph)

        best_state_data = (np.copy(state.slot_to_node), np.copy(state.core_assignment), np.copy(state.local_assignment))
        best_energy = current_energy

        T = T_start
        while T > T_min:
            for _ in range(steps_per_T):
                # Propose a random change
                undo_info = self._propose_mutation(state)
                new_energy = self._compute_energy(state, graph)
                delta_E = new_energy - current_energy

                # Acceptance criteria (Metropolis-Hastings)
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept change permanently
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state_data = (np.copy(state.slot_to_node), np.copy(state.core_assignment), np.copy(state.local_assignment))
                else:
                    # Reject and revert state
                    self._undo_mutation(state, undo_info)
            # Cool down
            T *= alpha
            print(f"Temp: {T:.2f} | Current Energy: {current_energy:.2f} | Best Energy: {best_energy:.2f}")

        # Restore best found configuration
        state.slot_to_node, state.core_assignment, state.local_assignment = best_state_data
        return state, best_energy

    def _compute_energy(self, state: SAState, graph: gt.Graph) -> int:
        # 1. compute optimal slice assignment
        clustering: ClusterAndHwOutput = ClusterAndHwOutput(hw=state.hw_config, num_clusters=state.hw_config.total_cores, cluster_assignment=state.core_assignment)
        opt_slice_assignment = OptimalSliceAssigner().assign_slices(clustering, graph, local_assignment=state.local_assignment)

        # 2. compute energy
        nr_inconsistencies: int = self._compute_inconsistencies(core_assignment=state.core_assignment, local_assignment=state.local_assignment,
                                                                opt_slice_assignment=opt_slice_assignment, hw=state.hw_config, graph=graph)
        return nr_inconsistencies

    def _compute_inconsistencies(self, core_assignment, local_assignment, opt_slice_assignment, hw: MosaicHardwareConfig, graph: gt.Graph) -> int:
        e_valid = 0
        for tgt in range(graph.num_vertices()):
            c_tgt = core_assignment[tgt]
            for src in graph.get_in_neighbors(tgt):
                c_src = core_assignment[src]
                dist = hw.core_distance(c_tgt, c_src)
                if dist == 0:
                    e_valid += 1
                else:
                    if hw.get_slice_bounds(dist, opt_slice_assignment[tgt][dist])[0] <= local_assignment[src] < hw.get_slice_bounds(dist, opt_slice_assignment[tgt][dist])[1]:
                        e_valid += 1
        return graph.num_edges() - e_valid

    def _propose_mutation(self, state: SAState):
        num_nodes = len(state.core_assignment)
        num_slots = state.slot_to_node.size

        # Decide between a Swap (0) or a Move to an empty slot (1)
        # If the hardware is 100% full, force a swap
        move_type = random.choice([SWAP, MOVE]) if num_slots > num_nodes else SWAP


        if move_type == SWAP:
            # SWAP: Pick two unique graph nodes
            node_a, node_b = random.sample(range(num_nodes), 2)
            # 1. Cache current coordinates
            core_a, local_a = state.core_assignment[node_a], state.local_assignment[node_a]
            core_b, local_b = state.core_assignment[node_b], state.local_assignment[node_b]

            # 2. Update the 1D assignment lookups
            state.core_assignment[node_a], state.core_assignment[node_b] = core_b, core_a
            state.local_assignment[node_a], state.local_assignment[node_b] = local_b, local_a

            # 3. Update the physical 2D grid matrix
            state.slot_to_node[core_a, local_a] = node_b
            state.slot_to_node[core_b, local_b] = node_a

            # Return undo information
            return SWAP, node_a, node_b
        else:
            # MOVE: Pick a node and a guaranteed empty slot
            node_a = random.randint(0, num_nodes - 1)
            old_core_a, old_local_a = state.core_assignment[node_a], state.local_assignment[node_a]

            empty_cores, empty_locals = np.where(state.slot_to_node == -1)
            random_idx = random.randrange(len(empty_cores))
            new_core = empty_cores[random_idx]
            new_local = empty_locals[random_idx]

            # apply move
            state.core_assignment[node_a] = new_core
            state.local_assignment[node_a] = new_local
            state.slot_to_node[old_core_a, old_local_a] = -1
            state.slot_to_node[new_core, new_local] = node_a

            # Apply move
            return MOVE, node_a, old_core_a, old_local_a

    def _undo_mutation(self, state: SAState, undo_info):
        if undo_info[0] == SWAP:
            # Unpack the exact state of both nodes before the swap occurred
            _, node_a, node_b = undo_info
            core_a, local_a = state.core_assignment[node_a], state.local_assignment[node_a]
            core_b, local_b = state.core_assignment[node_b], state.local_assignment[node_b]

            # 1. Revert 1D lookups back to their original coordinates
            state.core_assignment[node_a] = core_b
            state.local_assignment[node_a] = local_b

            state.core_assignment[node_b] = core_a
            state.local_assignment[node_b] = local_a

            # 2. Revert physical 2D grid occupancies
            state.slot_to_node[core_a, local_a] = node_b
            state.slot_to_node[core_b, local_b] = node_a

        elif undo_info[0] == MOVE:
            # Unpack the historical home of node_a
            _, node_a, old_core_a, old_local_a = undo_info

            # Identify where the node is currently sitting so we can vacate it
            current_core = state.core_assignment[node_a]
            current_local = state.local_assignment[node_a]

            # 1. Vacate the newly tried slot
            state.slot_to_node[current_core, current_local] = -1

            # 2. Revert 1D lookups back to the historical coordinates
            state.core_assignment[node_a] = old_core_a
            state.local_assignment[node_a] = old_local_a

            # 3. Restore node_a back to its original physical spot in the 2D grid
            state.slot_to_node[old_core_a, old_local_a] = node_a

