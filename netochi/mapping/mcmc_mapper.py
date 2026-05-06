"""
MCMC Mapper using Simulated Annealing via graph-tool's mcmc_anneal.

We inherit from graph-tool's MCMCState (which itself inherits from EntropyState)
and override `entropy()` and `mcmc_sweep()` with our hardware-specific
likelihood and move proposals.
"""

import threading
import graph_tool.inference.mcmc as gt_mcmc
from graph_tool.inference import MCMCState
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import Any

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


class HardwareMCMCState(MCMCState):
    """
    graph-tool compatible MCMC state for hardware mapping optimization.

    Inherits from MCMCState (-> EntropyState) and overrides:
      - entropy(): returns the negative log-likelihood of the Section 5 SBM model.
      - mcmc_sweep(): performs hardware-specific moves (node swap, slice mutation).
    """

    def __init__(self, mapping_state: MappingState, seed: int | None = None, verbose: bool = False):
        super().__init__(entropy_args={})
        self.state = mapping_state
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        
        # Best-state tracking so we can restore on timeout
        self._best_energy = float("inf")
        self._best_c: np.ndarray | None = None
        self._best_x: np.ndarray | None = None
        self._best_s: np.ndarray | None = None

    def entropy(self, **kwargs) -> float:
        """Return the energy (negative log-likelihood) to minimize."""
        if self.verbose:
            print(DEBUG_MCMC_ENTROPY_CALL)
        return -self.state.log_likelihood()

    def _save_best(self, energy: float) -> None:
        """Snapshot the current state if it is the best seen so far."""
        if energy < self._best_energy:
            self._best_energy = energy
            self._best_c = self.state.c.copy()
            self._best_x = self.state.x.copy()
            self._best_s = self.state.s.copy()

    def restore_best(self) -> None:
        """Restore the best-seen state (called after timeout or completion)."""
        if self.verbose:
            print(DEBUG_MCMC_RESTORE_BEST)
        if self._best_c is not None:
            self.state.c = self._best_c
            self.state.x = self._best_x
            self.state.s = self._best_s

    def mcmc_sweep(self, beta: float = 1.0, **kwargs) -> tuple:
        """Perform one sweep of N move-attempts with hardware-specific proposals."""
        if self.verbose:
            print(DEBUG_MCMC_SWEEP_CALL.format(beta=beta))
            
        nattempts = 0
        nmoves = 0
        delta_entropy = 0.0

        current_energy = self.entropy()
        self._save_best(current_energy)

        N = self.state.N
        for _ in range(N):
            nattempts += 1
            move_type = self.rng.integers(0, MCMC_NUM_MOVE_TYPES)
            node = int(self.rng.integers(0, N))

            if move_type == 0:
                # --- Node Swap ---
                node2 = int(self.rng.integers(0, N))
                if node == node2:
                    continue

                old_c1, old_x1 = self.state.c[node], self.state.x[node]
                old_c2, old_x2 = self.state.c[node2], self.state.x[node2]

                self.state.c[node], self.state.x[node] = old_c2, old_x2
                self.state.c[node2], self.state.x[node2] = old_c1, old_x1

                new_energy = self.entropy()
                dE = new_energy - current_energy

                if dE < 0 or self.rng.random() < np.exp(-dE * beta):
                    current_energy = new_energy
                    delta_entropy += dE
                    nmoves += 1
                    self._save_best(current_energy)
                else:
                    self.state.c[node], self.state.x[node] = old_c1, old_x1
                    self.state.c[node2], self.state.x[node2] = old_c2, old_x2

            else:
                # --- Slice Mutation ---
                d = int(self.rng.integers(1, self.state.config.max_distance + 1))
                n_slices = self.state.config.num_slices_at_distance(d)

                old_s = self.state.s[node, d]
                new_s = int(self.rng.integers(0, n_slices))

                if old_s == new_s:
                    continue

                self.state.s[node, d] = new_s
                new_energy = self.entropy()
                dE = new_energy - current_energy

                if dE < 0 or self.rng.random() < np.exp(-dE * beta):
                    current_energy = new_energy
                    delta_entropy += dE
                    nmoves += 1
                    self._save_best(current_energy)
                else:
                    self.state.s[node, d] = old_s

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
        """
        Run gt.mcmc_anneal with a hard wall-clock time limit.
        """
        if self.verbose:
            print(DEBUG_MCMC_RUN_START)
            
        graph = mapping_input.graph
        hw = mapping_input.hw_config

        # Build a mutable MappingState for the MCMC to mutate
        state = MappingState(graph, hw)
        state.init_random(seed=self.seed)

        hw_state = HardwareMCMCState(state, seed=self.seed, verbose=self.verbose)

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
                    verbose=False, # We handle verbose internally
                )
            except Exception as e:
                exc_holder.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.time_limit_s)

        if exc_holder:
            raise exc_holder[0]

        # Restore best mapping seen (even if timed out mid-sweep)
        hw_state.restore_best()

        return MosaicMappingState(
            mapping_input=mapping_input,
            neuron_core_idxs_assignment=state.c,
            neuron_local_idxs_assignment=state.x,
            neuron_slice_assignments=state.s,
        )
