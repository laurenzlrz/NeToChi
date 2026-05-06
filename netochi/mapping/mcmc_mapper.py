import threading
import graph_tool.inference.mcmc as gt_mcmc
from graph_tool.inference import MCMCState
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import Any, Optional

from netochi.mapping.likelihood_state import MappingState
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import BaseMapper, MosaicMappingState
from netochi.mapping.constants import (
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


class HardwareMCMCState(BaseModel, MCMCState):
    """
    Pydantic-based MCMC state for hardware mapping optimization.
    
    Inherits from MCMCState for graph-tool compatibility, but uses
    Pydantic for structural validation and configuration.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    # Inputs
    mapping_state: MappingState
    seed: Optional[int] = None
    verbose: bool = False

    # Private internal MCMC state
    _rng: np.random.Generator = PrivateAttr()
    _best_energy: float = PrivateAttr(default=float("inf"))
    _best_c: Optional[np.ndarray] = PrivateAttr(default=None)
    _best_x: Optional[np.ndarray] = PrivateAttr(default=None)
    _best_s: Optional[np.ndarray] = PrivateAttr(default=None)

    def __init__(self, **data):
        # Initialize BaseModel first
        super().__init__(**data)
        # Initialize MCMCState (EntropyState)
        MCMCState.__init__(self, entropy_args={})
        # Initialize private attributes
        self._rng = np.random.default_rng(self.seed)

    def entropy(self, **kwargs) -> float:
        """Return the energy to minimize."""
        if self.verbose:
            print(DEBUG_MCMC_ENTROPY_CALL)
        return -self.mapping_state.log_likelihood()

    def _save_best(self, energy: float) -> None:
        """Snapshot the current state if it is the best seen so far."""
        if energy < self._best_energy:
            self._best_energy = energy
            self._best_c = self.mapping_state.c.copy()
            self._best_x = self.mapping_state.x.copy()
            self._best_s = self.mapping_state.s.copy()

    def restore_best(self) -> None:
        """Restore the best-seen state (called after timeout or completion)."""
        if self.verbose:
            print(DEBUG_MCMC_RESTORE_BEST)
        if self._best_c is not None:
            self.mapping_state.c = self._best_c
            self.mapping_state.x = self._best_x
            self.mapping_state.s = self._best_s

    def mcmc_sweep(self, beta: float = 1.0, **kwargs) -> tuple:
        """Perform one sweep of N move-attempts."""
        if self.verbose:
            print(DEBUG_MCMC_SWEEP_CALL.format(beta=beta))
            
        nattempts = 0
        nmoves = 0
        delta_entropy = 0.0

        current_energy = self.entropy()
        self._save_best(current_energy)

        N = self.mapping_state.N
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
                d = int(self._rng.integers(1, self.mapping_state.config.max_distance + 1))
                n_slices = self.mapping_state.config.num_slices_at_distance(d)

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


class MCMCMapper(BaseModel, BaseMapper[MosaicMappingState, MosaicMappingInput[Any]]):
    """
    Pydantic-based MCMC Mapper using Simulated Annealing via graph-tool.
    """
    model_config = ConfigDict(frozen=True)

    iterations: int = Field(default=MCMC_DEFAULT_ITERATIONS)
    initial_temp: float = Field(default=MCMC_DEFAULT_INITIAL_TEMP)
    seed: int = Field(default=MCMC_DEFAULT_SEED)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S)
    verbose: bool = Field(default=False)

    def run(self, mapping_input: MosaicMappingInput[Any]) -> MosaicMappingState:
        """Run the optimization."""
        if self.verbose:
            print(DEBUG_MCMC_RUN_START)
            
        graph = mapping_input.graph
        hw = mapping_input.hw_config

        state = MappingState(graph, hw)
        state.init_random(seed=self.seed)

        # Initialize Pydantic-based MCMC state
        hw_state = HardwareMCMCState(
            mapping_state=state, 
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

        if exc_holder:
            raise exc_holder[0]

        hw_state.restore_best()

        return MosaicMappingState(
            mapping_input=mapping_input,
            neuron_core_idxs_assignment=state.c,
            neuron_local_idxs_assignment=state.x,
            neuron_slice_assignments=state.s,
        )
