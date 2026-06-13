import threading
import numpy as np
import numpy.typing as npt
import graph_tool.inference.mcmc as gt_mcmc
from graph_tool.inference import MCMCState
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import Optional, Generic, Tuple, List, Any, Dict

from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState, BaseMosaicMappingState
from netochi.objectives.interfaces import ObjectiveInterface
from netochi.definitions.constants import (
    MCMC_TIME_LIMIT_S,
    MCMC_DEFAULT_ITERATIONS,
    MCMC_DEFAULT_INITIAL_TEMP,
    MCMC_DEFAULT_SEED,
    MCMC_BETA_MULTIPLIER,
    MCMC_NUM_MOVE_TYPES,
    DEBUG_MCMC_RUN_START,
    DEBUG_MCMC_ENTROPY_CALL,
    DEBUG_MCMC_SWEEP_CALL,
    DEBUG_MCMC_RESTORE_BEST,
)


class HardwareMCMCState(MCMCState, Generic[PAYLOAD]):  # type: ignore[misc]
    """
    Internal MCMC state for hardware mapping optimization.
    """
    mapping_state: MosaicNetworkMappingState
    objective: ObjectiveInterface[BaseMosaicMappingState[Any]]
    seed: Optional[int]
    verbose: bool

    _rng: np.random.Generator
    _best_energy: float
    _best_c: Optional[npt.NDArray[np.int_]]
    _best_x: Optional[npt.NDArray[np.int_]]
    _best_s: Optional[npt.NDArray[np.int_]]

    def __init__(
        self, 
        mapping_state: MosaicNetworkMappingState,
        objective: ObjectiveInterface[BaseMosaicMappingState[Any]],
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        MCMCState.__init__(self)
        self.mapping_state = mapping_state
        self.objective = objective
        self.seed = seed
        self.verbose = verbose
        
        self._rng = np.random.default_rng(self.seed)
        self._best_energy = float("inf")
        self._best_c: Optional[npt.NDArray[np.int_]] = None
        self._best_x: Optional[npt.NDArray[np.int_]] = None
        self._best_s: Optional[npt.NDArray[np.int_]] = None

    def entropy(self, **kwargs: Any) -> float:
        """Return the energy to minimize."""
        if self.verbose:
            print(DEBUG_MCMC_ENTROPY_CALL)
        return -self.objective.log_likelihood(self.mapping_state)

    def _save_best(self, energy: float) -> None:
        """Snapshot the current state if it is the best seen so far."""
        if energy < self._best_energy:
            self._best_energy = energy
            self._best_c = self.mapping_state.c.copy()
            self._best_x = self.mapping_state.x.copy()
            self._best_s = self.mapping_state.s.copy()

    def restore_best(self) -> None:
        """Restore the best-seen state."""
        if self.verbose:
            print(DEBUG_MCMC_RESTORE_BEST)
        if self._best_c is not None and self._best_x is not None and self._best_s is not None:
            self.mapping_state.assignment = MosaicAssignment(
                hw=self.mapping_state.mapping_input.hw_config,
                neuron_core_pre_assignment=self._best_c.astype(np.int64),
                neuron_idx_pre_assignment=self._best_x.astype(np.int64),
                neuron_slice_assignment=self._best_s.astype(np.int64)
            )

    def mcmc_sweep(self, beta: float = 1.0, *args: Any, **kwargs: Any) -> Any:
        """Perform one sweep of N move-attempts."""
        if self.verbose:
            print(DEBUG_MCMC_SWEEP_CALL.format(beta=beta))
            
        nattempts = 0
        nmoves = 0
        delta_entropy = 0.0

        current_energy = self.entropy()
        self._save_best(current_energy)

        N: int = self.mapping_state.mapping_input.graph.num_vertices()
        for _ in range(N):
            nattempts += 1
            move_type = self._rng.integers(0, MCMC_NUM_MOVE_TYPES)
            node = int(self._rng.integers(0, N))

            if move_type == 0:
                # --- Node Swap ---
                node2 = int(self._rng.integers(0, N))
                if node == node2: continue

                old_c1, old_x1 = self.mapping_state.c[node], self.mapping_state.x[node]
                old_c2, old_x2 = self.mapping_state.c[node2], self.mapping_state.x[node2]

                self.mapping_state.c[node], self.mapping_state.x[node] = old_c2, old_x2
                self.mapping_state.c[node2], self.mapping_state.x[node2] = old_c1, old_x1

                new_energy = self.entropy()
                dE = new_energy - current_energy

                if dE < 0 or self._rng.random() < np.exp(-dE * beta):
                    current_energy = new_energy
                    delta_entropy += dE
                    nmoves += 1
                    self._save_best(current_energy)
                else:
                    self.mapping_state.c[node], self.mapping_state.x[node] = old_c1, old_x1
                    self.mapping_state.c[node2], self.mapping_state.x[node2] = old_c2, old_x2

            else:
                # --- Slice Mutation ---
                hw = self.mapping_state.mapping_input.hw_config
                d = int(self._rng.integers(1, hw.max_distance + 1))
                n_slices: int = hw.num_slices_at_distance(d)

                old_s = self.mapping_state.s[node, d]
                new_s = int(self._rng.integers(0, n_slices))

                if old_s == new_s: continue

                self.mapping_state.s[node, d] = new_s
                new_energy = self.entropy()
                dE = new_energy - current_energy

                if dE < 0 or self._rng.random() < np.exp(-dE * beta):
                    current_energy = new_energy
                    delta_entropy += dE
                    nmoves += 1
                    self._save_best(current_energy)
                else:
                    self.mapping_state.s[node, d] = old_s

        return delta_entropy, nattempts, nmoves

    def _get_entropy_args(self, *args: Any, **kwargs: Any) -> Any:
        return {}

    def _mcmc_sweep_dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return self.mcmc_sweep(*args, **kwargs)

    def multiflip_mcmc_sweep(self, *args: Any, **kwargs: Any) -> Any:
        return self.mcmc_sweep(*args, **kwargs)

    def gibbs_mcmc_sweep(self, *args: Any, **kwargs: Any) -> Any:
        return self.mcmc_sweep(*args, **kwargs)



class MCMCMapper(BaseModel, BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Pydantic-based MCMC Mapper using Simulated Annealing via graph-tool.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    objective: ObjectiveInterface[BaseMosaicMappingState[Any]]
    iterations: int = Field(default=MCMC_DEFAULT_ITERATIONS)
    initial_temp: float = Field(default=MCMC_DEFAULT_INITIAL_TEMP)
    seed: int = Field(default=MCMC_DEFAULT_SEED)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S)
    verbose: bool = Field(default=False)

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        """Run the optimization."""
        if self.verbose:
            print(DEBUG_MCMC_RUN_START)

        state: MosaicNetworkMappingState = MosaicNetworkMappingState.from_input_random(mapping_input, seed=self.seed)

        hw_state: HardwareMCMCState = HardwareMCMCState(
            mapping_state=state, 
            objective=self.objective,
            seed=self.seed, 
            verbose=self.verbose
        )

        beta_0 = 1.0 / self.initial_temp
        beta_1 = beta_0 * MCMC_BETA_MULTIPLIER

        mcmc_equilibrate_args = {
            "gibbs": False,
            "multiflip": False,
            "wait": 1,
            "force_niter": 1,
        }

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

        if exc_holder:
            raise Exception(repr(exc_holder[0]))

        hw_state.restore_best()

        return state
