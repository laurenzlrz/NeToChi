import threading
import math
import numpy as np
import numpy.typing as npt
import graph_tool.inference.mcmc as gt_mcmc
from graph_tool.inference import MCMCState
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import List, Tuple, Optional, Generic, Any, Dict

from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMapper, MosaicHWMappingState, BaseMosaicMappingState
from netochi.objectives.interfaces import LogLikelihoodObjectiveInterface
from netochi.mapping.mcmc.joint_inference_config import JointInferenceConfig
from netochi.mapping.mcmc.hardware_heuristic import MosaicHardwareHeuristic
from netochi.definitions.constants import (
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
    JOINT_P_NC,
    JOINT_P_NR,
    JOINT_P_L,
    JOINT_MAX_L,
    DEBUG_JOINT_NC_CHANGED,
    DEBUG_JOINT_NR_CHANGED,
    DEBUG_JOINT_L_CHANGED,
)


def log_star(n: int) -> float:
    """Universal prior for integers (log*)."""
    if n <= 0: return 0.0
    res: float = math.log(RISSANEN_C0) + math.log(n)
    curr: float = math.log(n)
    while curr > 1.0:
        curr = math.log(curr)
        res += curr
    return res


class JointHardwareMCMCState(MCMCState):
    """
    Internal MCMC state for JOINT hardware-mapping optimization.
    """
    mapping_state: MosaicHWMappingState[MappingInput]
    objective: LogLikelihoodObjectiveInterface[BaseMosaicMappingState[Any]]
    seed: Optional[int]
    verbose: bool
    K: int
    Nc: int
    Nr: int
    L: int
    config: JointInferenceConfig

    _rng: np.random.Generator
    _best_energy: float
    _best_K: int
    _best_Nc: int
    _best_Nr: int
    _best_L: int
    _best_c: Optional[npt.NDArray[np.int_]]
    _best_x: Optional[npt.NDArray[np.int_]]
    _best_s: Optional[npt.NDArray[np.int_]]

    def __init__(
        self, 
        mapping_state: MosaicHWMappingState[MappingInput],
        objective: LogLikelihoodObjectiveInterface[BaseMosaicMappingState[Any]],
        config: JointInferenceConfig,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        MCMCState.__init__(self)
        self.mapping_state = mapping_state
        self.objective = objective
        self.config = config
        self.seed = seed
        self.verbose = verbose

        self._rng = np.random.default_rng(self.seed)
        self.K = int(np.max(self.mapping_state.c) + 1)
        hw = self.mapping_state.inferred_hw
        self.Nc = hw.neurons_per_core
        self.Nr = hw.nodes_per_router
        self.L = hw.router_levels
        
        self._best_energy = float("inf")
        self._best_K = self.K
        self._best_Nc = self.Nc
        self._best_Nr = self.Nr
        self._best_L = self.L
        self._best_c: Optional[npt.NDArray[np.int_]] = None
        self._best_x: Optional[npt.NDArray[np.int_]] = None
        self._best_s: Optional[npt.NDArray[np.int_]] = None

    def mapping_cost(self, K: int) -> float:
        """Description length of the mapping given K cores."""
        num_neurons: int = self.mapping_state.mapping_input.graph.num_vertices()
        hw = self.mapping_state.inferred_hw
        Nc = hw.neurons_per_core
        cost_c = num_neurons * math.log(K)
        cost_x = num_neurons * math.log(Nc)
        cost_s = 0.0
        for d in range(1, hw.max_distance + 1):
            Sd = hw.num_slices_at_distance(d)
            cost_s += num_neurons * math.log(Sd)
        return cost_c + cost_x + cost_s

    def hardware_cost(self, K: int, Nc: int, Nr: int, L: int) -> float:
        """Physical hardware cost based on area and utilization."""
        # Silicon area cost for cores
        core_area_cost = self.config.alpha * (K * Nc)

        # Silicon area cost for routing fabric (assuming scaling with K * L * Nr)
        router_area_cost = self.config.beta * (K * L * Nr)

        # Utilization penalty (penalize wasted empty slots)
        num_neurons = self.mapping_state.mapping_input.graph.num_vertices()
        wasted_space = (K * Nc) - num_neurons
        utilization_penalty = self.config.gamma * max(0, wasted_space)

        return core_area_cost + router_area_cost + utilization_penalty

    def entropy(self, **kwargs: Any) -> float:
        """Total Description Length (Sigma)."""
        neg_ll = -self.objective.log_likelihood(self.mapping_state)
        c_map = self.mapping_cost(self.K)
        c_hw = self.hardware_cost(self.K, self.Nc, self.Nr, self.L)
        return neg_ll + c_map + c_hw

    def _save_best(self, energy: float) -> None:
        if energy < self._best_energy:
            self._best_energy = energy
            self._best_K = self.K
            self._best_Nc = self.Nc
            self._best_Nr = self.Nr
            self._best_L = self.L
            self._best_c = self.mapping_state.c.copy()
            self._best_x = self.mapping_state.x.copy()
            self._best_s = self.mapping_state.s.copy()

    def restore_best(self) -> None:
        if self._best_c is not None and self._best_x is not None and self._best_s is not None:
            self.K = self._best_K
            self.Nc = self._best_Nc
            self.Nr = self._best_Nr
            self.L = self._best_L
            self.mapping_state.hw_config_inferred = MosaicHardwareConfig(nodes_per_router=self.Nr, neurons_per_core=self.Nc, router_levels=self.L, slice_factor=2)
            self.mapping_state.neuron_core_idxs_assignment = self._best_c
            self.mapping_state.neuron_slice_assignments = self._best_s

    def _update_L(self) -> None:
        """Update router levels deterministically based on K and Nr."""
        # L = ceil(log_{Nr}(K))
        if self.Nr > 1:
            new_L = max(1, int(math.ceil(math.log(self.K) / math.log(self.Nr))))
        else:
            new_L = 1
            
        if new_L != self.L:
            old_L = self.L
            self.L = new_L
            self.mapping_state.hw_config_inferred = MosaicHardwareConfig(
                nodes_per_router=self.Nr, 
                neurons_per_core=self.Nc, 
                router_levels=self.L, 
                slice_factor=2
            )
            # If L increased, initialize new slice assignments for nodes
            if self.L > old_L:
                num_neurons = self.mapping_state.mapping_input.graph.num_vertices()
                for d in range(old_L + 1, self.L + 1):
                    n_sl = self.mapping_state.inferred_hw.num_slices_at_distance(d)
                    if n_sl > 0:
                        self.mapping_state.s[:, d] = self._rng.integers(0, n_sl, size=num_neurons)

    def mcmc_sweep(self, beta: float = 1.0, **kwargs: Any) -> Tuple[float, int, int]:
        """Perform one sweep with hardware size moves."""

        nattempts = 0
        nmoves = 0
        delta_entropy = 0.0
        current_energy = self.entropy()
        self._save_best(current_energy)

        num_neurons: int = self.mapping_state.mapping_input.graph.num_vertices()
        hw = self.mapping_state.inferred_hw
        
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
                    n_sl: int = hw.num_slices_at_distance(d)
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
                    if self.K >= (self.Nr ** self.L): continue
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
                    old_K, old_L = self.K, self.L
                    self.K += 1
                    self._update_L()
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                    else:
                        self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments, self.K, self.L = old_c, old_x, old_s, old_K, old_L
                        self.mapping_state.hw_config_inferred = MosaicHardwareConfig(nodes_per_router=self.Nr, neurons_per_core=self.Nc, router_levels=self.L, slice_factor=2)
                
                elif r < (JOINT_P_SWAP + JOINT_P_SLICE + JOINT_P_ADD_CORE + JOINT_P_REMOVE_CORE):
                    # --- REMOVE CORE ---
                    if self.K <= 1: continue
                    core_to_remove = int(self._rng.integers(0, self.K))
                    nodes_to_move = np.where(self.mapping_state.c == core_to_remove)[0]
                    
                    # Robust Reallocation: Check if other cores have enough space
                    core_counts = np.bincount(self.mapping_state.c, minlength=self.K)
                    total_available = np.sum(self.Nc - core_counts[np.arange(self.K) != core_to_remove])
                    if total_available < len(nodes_to_move): continue # Reject if no space
                    
                    old_c, old_x, old_s, old_K, old_L = self.mapping_state.c.copy(), self.mapping_state.x.copy(), self.mapping_state.s.copy(), self.K, self.L
                    
                    success = True
                    for node in nodes_to_move:
                        # Recalculate core counts to handle incremental moves
                        current_counts = np.bincount(self.mapping_state.c, minlength=self.K)
                        avail_cores = np.where((current_counts < self.Nc) & (np.arange(self.K) != core_to_remove))[0]
                        if len(avail_cores) == 0:
                            success = False; break
                        target_core = int(self._rng.choice(avail_cores))
                        x_in_target = self.mapping_state.x[self.mapping_state.c == target_core]
                        avail_x = [ix for ix in range(self.Nc) if ix not in x_in_target]
                        self.mapping_state.c[node] = target_core
                        self.mapping_state.x[node] = int(self._rng.choice(avail_x))
                    
                    if not success:
                        self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments = old_c, old_x, old_s
                        continue
                        
                    self.mapping_state.c[self.mapping_state.c > core_to_remove] -= 1
                    old_K, old_L = self.K, self.L
                    self.K -= 1
                    self._update_L()
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                    else:
                        self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments, self.K, self.L = old_c, old_x, old_s, old_K, old_L
                        self.mapping_state.hw_config_inferred = MosaicHardwareConfig(nodes_per_router=self.Nr, neurons_per_core=self.Nc, router_levels=self.L, slice_factor=2)

                elif r < (JOINT_P_SWAP + JOINT_P_SLICE + JOINT_P_ADD_CORE + JOINT_P_REMOVE_CORE + JOINT_P_NC):
                    # --- CHANGE NC ---
                    delta = 1 if self._rng.random() < 0.5 else -1
                    new_nc = self.Nc + delta
                    if new_nc < 1: continue
                    
                    old_c, old_x, old_s, old_nc, old_K, old_L = self.mapping_state.c.copy(), self.mapping_state.x.copy(), self.mapping_state.s.copy(), self.Nc, self.K, self.L
                    
                    # Capacity Check & Compensate
                    if num_neurons > (self.K * new_nc):
                        needed_K = int(math.ceil(num_neurons / new_nc))
                        if needed_K <= (self.Nr ** 10): # JOINT_MAX_L is 10
                            self.K = needed_K
                            self._update_L()
                        else:
                            continue # Physically impossible
                    
                    # Evict and Reallocate
                    over_nc_mask = self.mapping_state.x >= new_nc
                    nodes_to_evict = np.where(over_nc_mask)[0]
                    
                    if len(nodes_to_evict) > 0:
                        success = True
                        for node in nodes_to_evict:
                            current_counts = np.bincount(self.mapping_state.c, minlength=self.K)
                            avail_cores = np.where(current_counts < new_nc)[0]
                            if len(avail_cores) == 0:
                                success = False; break
                            target_core = int(self._rng.choice(avail_cores))
                            x_in_target = self.mapping_state.x[self.mapping_state.c == target_core]
                            avail_x = [ix for ix in range(new_nc) if ix not in x_in_target]
                            self.mapping_state.c[node] = target_core
                            self.mapping_state.x[node] = int(self._rng.choice(avail_x))
                        if not success:
                            self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments, self.K, self.L = old_c, old_x, old_s, old_K, old_L
                            continue
                    
                    self.Nc = new_nc
                    self.mapping_state.hw_config_inferred = MosaicHardwareConfig(nodes_per_router=self.Nr, neurons_per_core=self.Nc, router_levels=self.L, slice_factor=2)
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                    else:
                        self.mapping_state.neuron_core_idxs_assignment, self.mapping_state.neuron_local_idxs_assignment, self.mapping_state.neuron_slice_assignments, self.Nc, self.K, self.L = old_c, old_x, old_s, old_nc, old_K, old_L
                        self.mapping_state.hw_config_inferred = MosaicHardwareConfig(nodes_per_router=self.Nr, neurons_per_core=self.Nc, router_levels=self.L, slice_factor=2)
                
                elif r < (JOINT_P_SWAP + JOINT_P_SLICE + JOINT_P_ADD_CORE + JOINT_P_REMOVE_CORE + JOINT_P_NC + JOINT_P_NR):
                    # --- CHANGE NR ---
                    delta = 1 if self._rng.random() < 0.5 else -1
                    new_nr = self.Nr + delta
                    if new_nr < 1: continue
                    if self.K > new_nr ** self.L: continue
                    old_nr, old_L = self.Nr, self.L
                    self.Nr = new_nr
                    self._update_L()
                    new_energy = self.entropy()
                    dE = new_energy - current_energy
                    if dE < 0 or self._rng.random() < math.exp(-dE * beta):
                        current_energy, delta_entropy, nmoves = new_energy, delta_entropy + dE, nmoves + 1
                        self._save_best(current_energy)
                    else:
                        self.Nr, self.L = old_nr, old_L
                        self.mapping_state.hw_config_inferred = MosaicHardwareConfig(nodes_per_router=self.Nr, neurons_per_core=self.Nc, router_levels=self.L, slice_factor=2)
                
                # L is now implicit

        return delta_entropy, nattempts, nmoves
        
    def _get_entropy_args(self) -> Dict[str, Any]:
        return {}

    def _mcmc_sweep_dispatch(self, **kwargs: Any) -> Tuple[float, int, int]:
        return self.mcmc_sweep(**kwargs)

    def multiflip_mcmc_sweep(self, **kwargs: Any) -> Tuple[float, int, int]:
        return self.mcmc_sweep(**kwargs)

    def gibbs_mcmc_sweep(self, **kwargs: Any) -> Tuple[float, int, int]:
        return self.mcmc_sweep(**kwargs)


class JointInferenceMapper(BaseModel, BaseMapper[MosaicHWMappingState[MappingInput], MappingInput]):
    """
    Pydantic-based Joint Inference Mapper.
    Input is purely the network (MappingInput).
    Output is the state and the optimized hardware (MosaicHWMappingState).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    objective: LogLikelihoodObjectiveInterface[BaseMosaicMappingState[Any]]
    config: JointInferenceConfig = Field(default_factory=JointInferenceConfig)
    
    slice_factor: int = Field(default=2)
    
    iterations: int = Field(default=MCMC_DEFAULT_ITERATIONS)
    initial_temp: float = Field(default=MCMC_DEFAULT_INITIAL_TEMP)
    seed: int = Field(default=MCMC_DEFAULT_SEED)
    verbose: bool = Field(default=False)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S)
    
    def run(self, mapping_input: MappingInput) -> MosaicHWMappingState[MappingInput]:
        """Run joint inference by exploring the full hardware architecture."""
        # 1. Use heuristic to build initial hardware and placement
        state, init_hw = MosaicHardwareHeuristic.build_initial_hardware_state(
            mapping_input, 
            slice_factor=self.slice_factor,
            seed=self.seed
        )
        
        hw_state: JointHardwareMCMCState = JointHardwareMCMCState(
            mapping_state=state, 
            objective=self.objective,
            config=self.config,
            seed=self.seed, 
            verbose=self.verbose
        )

        beta_0 = 1.0 / self.initial_temp
        beta_1 = beta_0 * MCMC_BETA_MULTIPLIER

        mcmc_equilibrate_args = {"gibbs": False, "multiflip": False, "wait": 1, "force_niter": 1}
        exc_holder: List[Exception] = []

        def _run() -> None:
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
                import traceback
                traceback.print_exc()
                exc_holder.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.time_limit_s)

        if exc_holder: raise exc_holder[0]

        hw_state.restore_best()

        # 3. State already contains the optimized hw_config from restore_best

        return state

class MosaicHardwareMapper(BaseModel, BaseMapper[MosaicHWMappingState[MosaicMappingInput], MosaicMappingInput]):
    """
    A simple mapper that returns the ground truth hardware configuration from the input
    wrapped in a MosaicHWMappingState. Useful as a baseline for joint inference.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def run(self, mapping_input: MosaicMappingInput) -> MosaicHWMappingState[MosaicMappingInput]:
        """Returns the hardware from the input in a MosaicHWMappingState."""
        from netochi.mapping.random_mapper import RandomMapper
        random_state = RandomMapper().run(mapping_input)
        
        return MosaicHWMappingState(
            _mapping_input=mapping_input,
            _inferred_hw_config=mapping_input.hw_config,
            assignment=random_state.assignment
        )
