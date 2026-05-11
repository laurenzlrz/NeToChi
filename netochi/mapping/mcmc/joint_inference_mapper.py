import threading
import math
import numpy as np
import graph_tool.inference.mcmc as gt_mcmc
from graph_tool.inference import MCMCState
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import List, Tuple, Optional, Generic

from netochi.input_generator.interfaces import MappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMapper, MosaicHWMappingState, PAYLOAD
from netochi.objectives.log_likelihood import LogLikelihoodObjectiveInterface
from netochi.mapping.constants import (
    MCMC_TIME_LIMIT_S,
    MCMC_DEFAULT_ITERATIONS,
    MCMC_DEFAULT_INITIAL_TEMP,
    MCMC_DEFAULT_SEED,
    MCMC_BETA_MULTIPLIER,
    JOINT_NUM_MOVE_TYPES,
    JOINT_P_SWAP,
    JOINT_P_SLICE,
    JOINT_P_ADD_CORE,
    JOINT_P_REMOVE_CORE,
    JOINT_P_SPLIT,
    RISSANEN_C0,
    DEBUG_JOINT_RUN_START,
    DEBUG_JOINT_SWEEP_CALL,
    DEBUG_JOINT_CORE_ADDED,
    DEBUG_JOINT_CORE_REMOVED,
    DEBUG_JOINT_RESTORE_BEST,
)


def log_star(n: int) -> float:
    """Universal prior for integers (log*)."""
    if n <= 0: return 0.0
    res = math.log(RISSANEN_C0) + math.log(n)
    curr = math.log(n)
    while True:
        curr = math.log(curr)
        if curr <= 0: break
        res += curr
    return res


class JointHardwareMCMCState(BaseModel, MCMCState, Generic[PAYLOAD]):
    """
    Pydantic-based MCMC state for JOINT hardware-mapping optimization.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    # Unified HW State
    mapping_state: MosaicHWMappingState[MappingInput[PAYLOAD], PAYLOAD]
    objective: LogLikelihoodObjectiveInterface
    seed: Optional[int] = None
    verbose: bool = False

    # Mutable state
    K: int = Field(default=1)
    
    # Internal private MCMC state
    _rng: np.random.Generator = PrivateAttr()
    _K_max: int = PrivateAttr()
    _K_min: int = PrivateAttr()
    _best_energy: float = PrivateAttr(default=float("inf"))
    _best_K: int = PrivateAttr(default=1)
    _best_c: Optional[np.ndarray] = PrivateAttr(default=None)
    _best_x: Optional[np.ndarray] = PrivateAttr(default=None)
    _best_s: Optional[np.ndarray] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        MCMCState.__init__(self, entropy_args={})
        self._rng = np.random.default_rng(self.seed)
        self.K = int(np.max(self.mapping_state.c) + 1)
        hw = self.mapping_state.hw_config
        self._K_max = hw.total_cores
        num_neurons = self.mapping_state.mapping_input.graph.num_vertices()
        self._K_min = math.ceil(num_neurons / hw.neurons_per_core)

    def mapping_cost(self, K: int) -> float:
        """Description length of the mapping given K cores."""
        num_neurons = self.mapping_state.mapping_input.graph.num_vertices()
        hw = self.mapping_state.hw_config
        Nc = hw.neurons_per_core
        cost_c = num_neurons * math.log(K)
        cost_x = num_neurons * math.log(Nc)
        cost_s = 0.0
        for d in range(1, hw.max_distance + 1):
            Sd = hw.num_slices_at_distance(d)
            cost_s += num_neurons * math.log(Sd)
        return cost_c + cost_x + cost_s

    def hardware_cost(self, K: int) -> float:
        """Description length of the hardware configuration (K cores)."""
        return log_star(K)

    def entropy(self, **kwargs) -> float:
        """Total Description Length (Sigma)."""
        neg_ll = -self.objective.log_likelihood(self.mapping_state)
        c_map = self.mapping_cost(self.K)
        c_hw = self.hardware_cost(self.K)
        return neg_ll + c_map + c_hw

    def _save_best(self, energy: float) -> None:
        if energy < self._best_energy:
            self._best_energy = energy
            self._best_K = self.K
            self._best_c = self.mapping_state.c.copy()
            self._best_x = self.mapping_state.x.copy()
            self._best_s = self.mapping_state.s.copy()

    def restore_best(self) -> None:
        if self.verbose:
            print(DEBUG_JOINT_RESTORE_BEST.format(k=self._best_K, energy=self._best_energy))
        if self._best_c is not None:
            self.K = self._best_K
            self.mapping_state.neuron_core_idxs_assignment = self._best_c
            self.mapping_state.neuron_local_idxs_assignment = self._best_x
            self.mapping_state.neuron_slice_assignments = self._best_s

    def mcmc_sweep(self, beta: float = 1.0, **kwargs) -> tuple:
        """Perform one sweep with hardware size moves."""
        if self.verbose:
            print(DEBUG_JOINT_SWEEP_CALL.format(beta=beta, k=self.K))

        nattempts = 0
        nmoves = 0
        delta_entropy = 0.0
        current_energy = self.entropy()
        self._save_best(current_energy)

        num_neurons = self.mapping_state.mapping_input.graph.num_vertices()
        hw = self.mapping_state.hw_config
        
        for _ in range(num_neurons):
            nattempts += 1
            r = self._rng.random()
            
            if r < (JOINT_P_SWAP + JOINT_P_SLICE):
                if r < JOINT_P_SWAP:
                    # Node Swap
                    node1 = int(self._rng.integers(0, num_neurons))
                    node2 = int(self._rng.integers(0, num_neurons))
                    if node1 == node2: continue
                    old_c1, old_x1 = self.mapping_state.c[node1], self.mapping_state.x[node1]
                    old_c2, old_x2 = self.mapping_state.c[node2], self.mapping_state.x[node2]
                    self.mapping_state.c[node1], self.mapping_state.x[node1] = old_c2, old_x2
                    self.mapping_state.c[node2], self.mapping_state.x[node2] = old_c1, old_x1
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                    else:
                        self.mapping_state.c[node1], self.mapping_state.x[node1] = old_c1, old_x1
                        self.mapping_state.c[node2], self.mapping_state.x[node2] = old_c2, old_x2
                else:
                    # Slice Mutation
                    node = int(self._rng.integers(0, num_neurons))
                    d = int(self._rng.integers(1, hw.max_distance + 1))
                    n_sl = hw.num_slices_at_distance(d)
                    old_s = self.mapping_state.s[node, d]
                    new_s = int(self._rng.integers(0, n_sl))
                    if old_s == new_s: continue
                    self.mapping_state.s[node, d] = new_s
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                    else:
                        self.mapping_state.s[node, d] = old_s

            else:
                if r < (JOINT_P_SWAP + JOINT_P_SLICE + JOINT_P_ADD_CORE):
                    # --- ADD CORE ---
                    if self.K >= self._K_max: continue
                    core_to_split = int(self._rng.integers(0, self.K))
                    nodes_in_core = np.where(self.mapping_state.c == core_to_split)[0]
                    if len(nodes_in_core) <= 1: continue
                    old_c, old_x, old_s = self.mapping_state.c.copy(), self.mapping_state.x.copy(), self.mapping_state.s.copy()
                    moved_mask = self._rng.random(len(nodes_in_core)) < JOINT_P_SPLIT
                    moved_nodes = nodes_in_core[moved_mask]
                    if len(moved_nodes) == 0 or len(moved_nodes) == len(nodes_in_core): continue
                    self.mapping_state.c[moved_nodes] = self.K
                    avail_x = list(range(hw.neurons_per_core))
                    self._rng.shuffle(avail_x)
                    self.mapping_state.x[moved_nodes] = avail_x[:len(moved_nodes)]
                    for d in range(1, hw.max_distance + 1):
                        n_sl = hw.num_slices_at_distance(d)
                        self.mapping_state.s[moved_nodes, d] = self._rng.integers(0, n_sl, size=len(moved_nodes))
                    old_K = self.K
                    self.K += 1
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                        if self.verbose: print(DEBUG_JOINT_CORE_ADDED.format(k_old=old_K, k_new=self.K))
                    else:
                        self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments, self.K = old_c, old_x, old_s, old_K
                
                else:
                    # --- REMOVE CORE ---
                    if self.K <= self._K_min: continue
                    core_to_remove = int(self._rng.integers(0, self.K))
                    nodes_to_move = np.where(self.mapping_state.c == core_to_remove)[0]
                    old_c, old_x, old_s = self.mapping_state.c.copy(), self.mapping_state.x.copy(), self.mapping_state.s.copy()
                    success = True
                    for node in nodes_to_move:
                        core_counts = np.bincount(self.mapping_state.c, minlength=self.K)
                        avail_cores = np.where((core_counts < hw.neurons_per_core) & 
                                              (np.arange(self.K) != core_to_remove))[0]
                        if len(avail_cores) == 0:
                            success = False; break
                        target_core = int(self._rng.choice(avail_cores))
                        self.mapping_state.c[node] = target_core
                        x_in_target = self.mapping_state.x[self.mapping_state.c == target_core]
                        avail_x = [ix for ix in range(hw.neurons_per_core) if ix not in x_in_target]
                        self.mapping_state.x[node] = int(self._rng.choice(avail_x))
                        for d in range(1, hw.max_distance + 1):
                            n_sl = hw.num_slices_at_distance(d)
                            self.mapping_state.s[node, d] = self._rng.integers(0, n_sl)
                    if not success:
                        self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments = old_c, old_x, old_s
                        continue
                    self.mapping_state.c[self.mapping_state.c > core_to_remove] -= 1
                    old_K = self.K
                    self.K -= 1
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                        if self.verbose: print(DEBUG_JOINT_CORE_REMOVED.format(k_old=old_K, k_new=self.K))
                    else:
                        self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments, self.K = old_c, old_x, old_s, old_K

        return delta_entropy, nattempts, nmoves


class JointInferenceMapper(BaseModel, Generic[PAYLOAD], BaseMapper[MosaicHWMappingState[MappingInput[PAYLOAD], PAYLOAD], MappingInput[PAYLOAD]]):
    """
    Pydantic-based Joint Inference Mapper.
    Input is purely the network (MappingInput).
    Output is the state and the optimized hardware (MosaicHWMappingState).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    objective: LogLikelihoodObjectiveInterface
    hw_template: MosaicHardwareConfig  # Template constraints for the hardware (e.g. max_distance, neurons_per_core)
    
    iterations: int = Field(default=MCMC_DEFAULT_ITERATIONS)
    initial_temp: float = Field(default=MCMC_DEFAULT_INITIAL_TEMP)
    seed: int = Field(default=MCMC_DEFAULT_SEED)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S)
    verbose: bool = Field(default=False)

    def run(self, mapping_input: MappingInput[PAYLOAD]) -> MosaicHWMappingState[MappingInput[PAYLOAD], PAYLOAD]:
        """Run joint inference starting from the template hardware constraints."""
        if self.verbose:
            k_min = math.ceil(mapping_input.graph.num_vertices() / self.hw_template.neurons_per_core)
            print(DEBUG_JOINT_RUN_START.format(k_min=k_min, k_max=self.hw_template.total_cores))

        state = MosaicHWMappingState.from_input_and_hw(mapping_input, self.hw_template)
        state.init_random_assignments(seed=self.seed)

        hw_state = JointHardwareMCMCState(
            mapping_state=state, 
            objective=self.objective,
            seed=self.seed, 
            verbose=self.verbose
        )

        beta_0 = 1.0 / self.initial_temp
        beta_1 = beta_0 * MCMC_BETA_MULTIPLIER

        mcmc_equilibrate_args = {"gibbs": False, "multiflip": False, "wait": 1, "force_niter": 1}
        exc_holder: list = []

        def _run():
            try:
                gt_mcmc.mcmc_anneal(
                    hw_state,
                    beta_range=(beta_0, beta_1),
                    niter=self.iterations,
                    mcmc_equilibrate_args=mcmc_equilibrate_args,
                    history=False,
                    verbose=False,
                )
            except Exception as e:
                exc_holder.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.time_limit_s)

        if exc_holder: raise exc_holder[0]

        hw_state.restore_best()

        optimized_hw = MosaicHardwareConfig(
            neurons_per_core=self.hw_template.neurons_per_core,
            total_cores=hw_state.K,
            router_levels=self.hw_template.router_levels,
            slices_per_level=self.hw_template.slices_per_level,
            max_distance=self.hw_template.max_distance
        )
        state.hw_config = optimized_hw

        return state
