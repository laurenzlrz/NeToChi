import math
import numpy as np
import graph_tool.all as gt
from typing import Any, Optional, Tuple
import icontract

from netochi.input_generator.interfaces import MappingInput, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMapper, MosaicHWMappingState
from netochi.mapping.sa_ihw_config import SimAnnealingIHWConfig
from netochi.mapping.simulated_annealing_inferred_hw.sa_ihw_state import SAIHWState
from netochi.mapping.simulated_annealing_inferred_hw.sa_ihw_mutation import (
    IHWMutation,
    IHWSwapMutation,
    IHWMoveMutation,
    IHWAddCoreMutation,
    IHWIncrementNcMutation,
    IHWSwapCoresMutation,
    IHWRemoveCoreMutation,
    IHWDecrementNcMutation
)
from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner
from netochi.objectives.utils import compute_e_valid


class SimAnnealingInferredHWMapper(BaseMapper[MosaicHWMappingState[MappingInput], MappingInput]):
    """
    Simulated Annealing Inferred Hardware Mapper.
    Input is purely the network (MappingInput).
    Output is the state and the optimized hardware (MosaicHWMappingState).
    """

    @icontract.require(lambda config: config is None or isinstance(config, SimAnnealingIHWConfig))
    def __init__(self, config: Optional[SimAnnealingIHWConfig] = None) -> None:
        self.config = config or SimAnnealingIHWConfig()
        self._state: Optional[SAIHWState] = None
        self._opt_slice_assigner: Optional[DeltaOptimalSliceAssigner] = None
        self._graph: Optional[gt.Graph] = None
        self._rng: np.random.Generator = np.random.default_rng()
        self._in_edges_data: dict[str, Any] = {}


    @classmethod
    def _find_initial_hardware_config(cls, graph: gt.Graph, slice_factor: int) -> MosaicHardwareConfig:
        N = graph.num_vertices()
        if N > 0:
            D = (1.0 * graph.num_edges()) / N
        else:
            D = 0.0

        margin = 3.0
        nc_init = int(max(4, min(256, math.ceil(D * margin))))
        K_init = int(math.ceil(N / nc_init))
        nr_init = 4 if D < 4 else 8

        if nr_init > 1:
            l_init = int(max(1, math.ceil(math.log(K_init) / math.log(nr_init))))
        else:
            l_init = 1

        return MosaicHardwareConfig(
            nodes_per_router=nr_init,
            neurons_per_core=nc_init,
            router_levels=l_init,
            slice_factor=slice_factor
        )

    def run(self, mapping_input: MappingInput) -> MosaicHWMappingState[MappingInput]:
        self._graph = mapping_input.graph
        self._rng = np.random.default_rng(self.config.seed)

        init_hw = self._find_initial_hardware_config(mapping_input.graph, self.config.slice_factor)
        N = self._graph.num_vertices()
        init_assignment = MosaicAssignment.random(N, init_hw, seed=self.config.seed)

        # 2. Build SAIHWState
        self._state = SAIHWState(
            mapping_input=mapping_input,
            initial_hw=init_hw,
            assignment=init_assignment
        )
        self._state.unfreeze()

        # 3. Build DeltaOptimalSliceAssigner and link it to state
        self._opt_slice_assigner = DeltaOptimalSliceAssigner(
            hw_config=init_hw,
            graph=mapping_input.graph,
            cluster_assignment=self._state.c,
            local_assignment=self._state.x
        )
        self._state._slice_assigner = self._opt_slice_assigner

        # Precompute in-edges to optimize compute_e_valid performance
        self._in_edges_data = {
            'N': self._graph.num_vertices(),
            'in_edges': [list(self._graph.get_in_neighbors(tgt)) for tgt in range(self._graph.num_vertices())]
        }

        # 4. Run simulated annealing
        best_assignment: MosaicAssignment = self._run_simulated_annealing()

        self._state.update_assignment(best_assignment)

        # 5. Return MosaicHWMappingState
        self._state.freeze()

        return MosaicHWMappingState(
            mapping_input=mapping_input,
            inferred_hw=self._state.hw_config,
            assignment=self._state.assignment
        )

    def _run_simulated_annealing(self) -> MosaicAssignment:
        assert self._graph is not None
        assert self._state is not None
        assert self._opt_slice_assigner is not None

        import time
        start_time = time.time()

        steps_per_T = self.config.steps_per_T
        if steps_per_T is None:
            steps_per_T = 10 * self._graph.num_vertices()

        current_energy = self._compute_energy()
        best_energy = current_energy

        # Save copies of the initial best state arrays to avoid Pydantic object allocation overhead in the loop
        best_hw = self._state.hw_config
        best_c = self._state.c.copy()
        best_x = self._state.x.copy()
        best_s = self._opt_slice_assigner.slice_assignment.copy()

        T = self.config.T_start
        while T > self.config.T_min:
            if self.config.time_limit is not None and (time.time() - start_time) > self.config.time_limit:
                if self.config.verbose:
                    print("Time limit reached. Terminating simulated annealing.")
                break

            for _ in range(steps_per_T):
                # Propose a random mutation
                mutation = self._do_mutation()
                if mutation is None:
                    print("Relict, should not happen")
                    continue

                # Apply mutation (may return a new assigner if hardware changed)
                self._opt_slice_assigner = mutation.do(self._state, self._opt_slice_assigner, self._graph)
                new_energy = self._compute_energy()
                delta_E = new_energy - current_energy
                rndm = self._rng.random()

                if delta_E < 0 or rndm < math.exp(-delta_E / T):
                    # Accept mutation permanently
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_hw = self._state.hw_config
                        best_c = self._state.c.copy()
                        best_x = self._state.x.copy()
                        best_s = self._opt_slice_assigner.slice_assignment.copy()
                else:
                    # Reject and undo mutation (restores the old assigner if hardware changed)
                    self._opt_slice_assigner = mutation.undo(self._state, self._opt_slice_assigner)

            # Cool down
            T *= self.config.alpha_temp
            if self.config.verbose:
                print(f"Temp: {T:.4f} | Current Energy: {current_energy:.4f} | Best Energy: {best_energy:.2f}")

        # Construct MosaicAssignment only once at the end
        return MosaicAssignment(
            hw=best_hw,
            neuron_core_pre_assignment=best_c,
            neuron_idx_pre_assignment=best_x,
            neuron_slice_assignment=best_s
        )

    def _compute_hardware_cost(self, K: int, Nc: int, Nr: int, L: int) -> float:
        """Physical hardware cost based on area and utilization."""
        core_area_cost = self.config.alpha * (K * Nc)
        router_area_cost = self.config.beta * (K * L * Nr)
        num_neurons = self._graph.num_vertices() if self._graph is not None else 0
        wasted_space = (K * Nc) - num_neurons
        utilization_penalty = self.config.gamma * max(0, wasted_space)
        return core_area_cost + router_area_cost + utilization_penalty

    def _compute_energy(self) -> float:
        assert self._state is not None
        assert self._opt_slice_assigner is not None
        assert self._graph is not None

        # Compute inconsistencies using the official utilities function
        e_valid = compute_e_valid(self._state, self._in_edges_data)
        inconsistencies = self._graph.num_edges() - e_valid

        # Compute hardware cost
        hw_cost = self._compute_hardware_cost(self._state.K, self._state.Nc, self._state.Nr, self._state.L)

        return self.config.weight_inconsistencies * inconsistencies + hw_cost

    def _do_mutation(self) -> Optional[IHWMutation]:
        assert self._state is not None
        assert self._rng is not None

        mutation_types = [
            IHWSwapMutation,
            IHWMoveMutation,
            IHWAddCoreMutation,
            IHWRemoveCoreMutation,
            IHWIncrementNcMutation,
            IHWDecrementNcMutation,
            IHWSwapCoresMutation
        ]
        probs = [
            self.config.p_swap,
            self.config.p_move,
            self.config.p_add_core,
            self.config.p_remove_core,
            self.config.p_increment_nc,
            self.config.p_decrement_nc,
            self.config.p_swap_cores
        ]
        total = sum(probs)
        normalized_probs = [p / total for p in probs]

        # Retry up to 10 times to select a valid mutation
        for _ in range(10):
            mutation_cls = self._rng.choice(mutation_types, p=normalized_probs)
            mutation = mutation_cls.create_random(self._state, self._rng)
            if mutation is not None:
                return mutation
        return None
