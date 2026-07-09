import math
import random
from typing import Any, List
import numpy as np
import numpy.typing as npt
import graph_tool.all as gt
from pydantic import BaseModel, Field

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.simulated_annealing_fix_hw.sa_mutation import SwapMutation, MoveMutation
from netochi.mapping.simulated_annealing_fix_hw.sa_state import LocalAddressSAState
from netochi.mapping.simulated_annealing_mapper import compute_inconsistencies
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput
from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner

SWAP = 0
MOVE = 1


class SALocalAddressAssignerConfig(BaseModel):
    T_start: float = Field(default=10.0)
    T_min: float = Field(default=0.1)
    alpha: float = Field(default=0.98)
    steps_per_T_multiplier: int = Field(default=2)
    verbose: bool = Field(default=False)

    def create(self) -> "SALocalAddressAssigner":
        return SALocalAddressAssigner(config=self)




class SALocalAddressAssigner(LocalAddressAssigner):
    """
    Assigns local addresses using Simulated Annealing, keeping core assignments strictly fixed.
    """

    def __init__(self, config: SALocalAddressAssignerConfig):
        self.config = config

    def assign_addresses(self, mapping_input: MosaicMappingInput, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:
        hw = mapping_input.hw_config
        graph = mapping_input.graph
        num_nodes = graph.num_vertices()
        core_assignment = clustering.cluster_assignment.astype(np.int_)

        # --- 1. Initialize State ---
        local_assignment = np.zeros(num_nodes, dtype=np.int_)
        slot_to_node = np.full((hw.total_cores, hw.neurons_per_core), -1, dtype=np.int_)

        # Group nodes by core to ensure mutations stay intra-core
        nodes_by_core: List[List[int]] = [[] for _ in range(hw.total_cores)]
        for node in range(num_nodes):
            c = core_assignment[node]
            nodes_by_core[c].append(node)

        # Create an initial valid random local assignment
        assert len(nodes_by_core[c]) <= hw.neurons_per_core, f"Core {c} is overloaded!"
        for c in range(hw.total_cores):
            if not nodes_by_core[c]:
                continue
            available_locals = list(range(hw.neurons_per_core))
            random.shuffle(available_locals)
            for node in nodes_by_core[c]:
                loc = available_locals.pop()
                local_assignment[node] = loc
                slot_to_node[c, loc] = node

        state = LocalAddressSAState(core_assignment.copy(), local_assignment, slot_to_node, hw)

        opt_slice_assigner = DeltaOptimalSliceAssigner(
            hw_config=hw,
            graph=graph,
            cluster_assignment=state.core_assignment,
            local_assignment=state.local_assignment
        )

        # --- 2. Run Simulated Annealing ---
        best_local_assignment = self._run_simulated_annealing(state, opt_slice_assigner, graph, nodes_by_core)

        return best_local_assignment

    def _run_simulated_annealing(self, state: LocalAddressSAState, opt_slice_assigner: DeltaOptimalSliceAssigner,
                                 graph: gt.Graph, nodes_by_core: List[List[int]]) -> npt.NDArray[np.int_]:

        T = self.config.T_start
        steps_per_T = self.config.steps_per_T_multiplier * graph.num_vertices()
        hw = state.hw_config

        current_energy = compute_inconsistencies(state.core_assignment, state.local_assignment,
                                                       opt_slice_assigner.slice_assignment, hw, graph)
        best_energy = current_energy
        best_local = state.local_assignment.copy()

        # Only consider cores that actually have nodes in them
        active_cores = [c for c in range(hw.total_cores) if len(nodes_by_core[c]) > 0]

        while T > self.config.T_min:
            for _ in range(steps_per_T):
                # --- Propose Intra-Core Mutation ---
                # 1. Pick a random active core
                c = random.choice(active_cores)
                core_nodes = nodes_by_core[c]
                num_core_nodes = len(core_nodes)

                # 2. Determine available moves within THIS core
                empty_locals = np.where(state.slot_to_node[c] == -1)[0]
                can_swap = num_core_nodes >= 2
                can_move = len(empty_locals) > 0

                if not can_swap and not can_move:
                    continue  # Edge case: Core is size 1 and full

                if can_swap and can_move:
                    move_type = random.choice([SWAP, MOVE])
                elif can_swap:
                    move_type = SWAP
                else:
                    move_type = MOVE

                # 3. Instantiate mutation
                if move_type == SWAP:
                    node_a, node_b = random.sample(core_nodes, 2)
                    mutation = SwapMutation(node_a, node_b)
                else:
                    node = random.choice(core_nodes)
                    new_local = random.choice(empty_locals)
                    # Notice we pass 'c' (the CURRENT core), not a random new core
                    mutation = MoveMutation(node, c, new_local)

                # Apply mutation
                mutation.do(state, opt_slice_assigner, graph)

                # --- Energy Check ---
                # NOTE: If you implemented the DeltaInconsistencyComputer, plug it in here instead of the O(|E|) full compute!
                new_energy = compute_inconsistencies(state.core_assignment, state.local_assignment,
                                                           opt_slice_assigner.slice_assignment, hw, graph)
                delta_E = new_energy - current_energy

                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_local = state.local_assignment.copy()
                else:
                    # Reject and rollback
                    mutation.undo(state, opt_slice_assigner)

            T *= self.config.alpha
            if self.config.verbose:
                print(f"Temp: {T:.2f} | Current Energy: {current_energy:.2f} | Best Energy: {best_energy:.2f}")

        return best_local
