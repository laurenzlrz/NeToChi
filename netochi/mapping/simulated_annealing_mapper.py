import math
import random
from typing import Any, Optional
import numpy.typing as npt

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment

import numpy as np
import graph_tool.all as gt
from dataclasses import dataclass

from netochi.mapping.simulated_annealing_fix_hw.sa_mutation import SAState, Mutation, SwapMutation, MoveMutation
from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner


SWAP = 0
MOVE = 1


@dataclass
class BestStateData:
    core_assignment: npt.NDArray[np.int64]
    local_assignment: npt.NDArray[np.int64]
    slice_assignment: npt.NDArray[np.int64]


class SimAnnealingMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper using simulated annealing. Takes hw config as input.

    State: (injective) mapping neuron_id -> (core_id, local_address)
    Mutation: Swap two neurons or move neuron to empty slot
    Loss: Given a mapping, we can compute the optimal slice assignment efficiently. We can then compute the number of inconsistencies.
    """

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        self.opt_slice_assigner: Optional[DeltaOptimalSliceAssigner] = None
        self.state: Optional[SAState] = None
        self.graph: Optional[gt.Graph] = None
        self.verbose: Optional[bool] = False

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        # --- 1. initialize ---
        self.state = SAState(mapping_input)
        self.opt_slice_assigner = DeltaOptimalSliceAssigner(hw_config=mapping_input.hw_config, graph=mapping_input.graph,
                                                            cluster_assignment=self.state.core_assignment, local_assignment=self.state.local_assignment)
        self.graph = mapping_input.graph

        # --- 2. run simulated annealing ---
        best_state_data: BestStateData = self._run_simulated_annealing()

        return MosaicNetworkMappingState(_mapping_input=mapping_input,
                                         assignment=MosaicAssignment(
                                            neuron_idx_pre_assignment=best_state_data.local_assignment.astype(np.int64),
                                            neuron_core_pre_assignment=best_state_data.core_assignment.astype(np.int64),
                                            neuron_slice_assignment=best_state_data.slice_assignment.astype(np.int64),
                                            hw=mapping_input.hw_config),
                                         )



    def _run_simulated_annealing(self, T_start=50.0, T_min=0.01, alpha=0.98, steps_per_T=100) -> BestStateData: # gemini: T_start = 10000, alpha=0.98
        """
        invariant: slice assigner is always in same state as SA state
        """
        assert self.graph is not None
        assert self.state is not None
        assert self.opt_slice_assigner is not None

        if steps_per_T is None:
            steps_per_T = 10 * self.graph.num_vertices()

        current_energy = self._compute_energy()
        best_energy = current_energy
        best_state_data: BestStateData = BestStateData(core_assignment = self.state.core_assignment.copy(),
                                                       local_assignment=self.state.local_assignment.copy(),
                                                       slice_assignment=self.opt_slice_assigner.slice_assignment.copy())

        T = T_start
        while T > T_min:
            for _ in range(steps_per_T):
                # Propose a random mutation
                mutation: Mutation = self._do_mutation()
                new_energy = self._compute_energy()
                delta_E = new_energy - current_energy

                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept mutation permanently
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state_data: BestStateData = BestStateData(core_assignment=self.state.core_assignment.copy(),
                                                                       local_assignment=self.state.local_assignment.copy(),
                                                                       slice_assignment=self.opt_slice_assigner.slice_assignment.copy())

                else:
                    # Reject and undo mutation
                    mutation.undo(state=self.state, slice_assigner=self.opt_slice_assigner)
            # Cool down
            T *= alpha
            if self.verbose:
                print(f"Temp: {T:.2f} | Current Energy: {current_energy:.2f} | Best Energy: {best_energy:.2f}")

        return best_state_data

    def _compute_energy(self) -> int:
        assert self.state is not None
        assert self.opt_slice_assigner is not None
        assert self.graph is not None
        return self._compute_inconsistencies(core_assignment=self.state.core_assignment,
                                             local_assignment=self.state.local_assignment,
                                             opt_slice_assignment=self.opt_slice_assigner.slice_assignment,
                                             hw=self.state.hw_config,
                                             graph=self.graph)

    def _do_mutation(self) -> Mutation:
        assert self.state is not None
        assert self.opt_slice_assigner is not None
        assert self.graph is not None
        num_nodes = len(self.state.core_assignment)
        num_slots = self.state.slot_to_node.size

        # Decide between a Swap (0) or a Move to an empty slot (1)
        # If the hardware is 100% full, force a swap
        move_type = random.choice([SWAP, MOVE]) if num_slots > num_nodes else SWAP

        if move_type == SWAP:
            # SWAP: Pick two unique graph nodes
            node_a, node_b = random.sample(range(num_nodes), 2)
            swap_mutation = SwapMutation(node_a, node_b)
            swap_mutation.do(state=self.state, slice_assigner=self.opt_slice_assigner, graph=self.graph)
            return swap_mutation
        else:
            # MOVE: Pick a node and a guaranteed empty slot
            node: int = random.randint(0, num_nodes - 1)
            empty_cores, empty_locals  = np.where(self.state.slot_to_node == -1)
            random_idx: int = random.randrange(len(empty_cores))
            new_core: int = empty_cores[random_idx].item()
            new_local_address: int = empty_locals[random_idx].item()

            move_mutation = MoveMutation(node, new_core, new_local_address)
            move_mutation.do(state=self.state, slice_assigner=self.opt_slice_assigner, graph=self.graph)
            return move_mutation


    def _compute_inconsistencies(self, core_assignment, local_assignment, opt_slice_assignment, hw: MosaicHardwareConfig, graph: gt.Graph) -> int:
        # TODO FUTURE USE EXISTING FUNCTION
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