import time
import threading
import graph_tool.inference.mcmc as gt_mcmc
import numpy as np
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.core import FixedHardwareInput

_MCMC_TIME_LIMIT_S = 10.0


class GraphToolCompatibleState:
    """Wrapper to make MappingState compatible with gt.mcmc_anneal."""

    def __init__(self, mapping_state: MappingState, objective, seed=None):
        self.state = mapping_state
        self.objective = objective
        self.rng = np.random.default_rng(seed)
        # Best-state tracking so we can restore on timeout
        self._best_energy = float("inf")
        self._best_c: np.ndarray | None = None
        self._best_x: np.ndarray | None = None
        self._best_s: np.ndarray | None = None

    def entropy(self, **kwargs) -> float:
        """Graph-tool MCMC minimizes entropy. We route that to our cost function."""
        if self.objective:
            return self.objective.evaluate(self.state)
        return -self.state.log_likelihood()

    def _save_best(self, energy: float) -> None:
        if energy < self._best_energy:
            self._best_energy = energy
            self._best_c = self.state.c.copy()
            self._best_x = self.state.x.copy()
            self._best_s = self.state.s.copy()

    def restore_best(self) -> None:
        """Restore the best-seen state (called after timeout or completion)."""
        if self._best_c is not None:
            self.state.c = self._best_c
            self.state.x = self._best_x
            self.state.s = self._best_s

    def mcmc_sweep(self, beta: float = 1.0, **kwargs) -> tuple:
        """
        Perform one sweep of N move-attempts.
        beta is the inverse temperature provided by gt.mcmc_anneal.
        Returns (delta_entropy, nattempts, nmoves).
        """
        nattempts = 0
        nmoves = 0
        delta_entropy = 0.0

        current_energy = self.entropy()
        self._save_best(current_energy)

        N = self.state.N
        for _ in range(N):
            nattempts += 1
            move_type = self.rng.integers(0, 2)
            node = int(self.rng.integers(0, N))

            if move_type == 0:
                # Swap core and local address between two nodes
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
                # Mutate a slice assignment
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


class Section5SBM_MCMCMapper:
    """Mapper using Simulated Annealing (via Graph-Tool) limited to 10 seconds."""
    
    def get_name(self) -> str:
        return self.__class__.__name__

    def map_fixed_hardware(
        self,
        mapping_input: FixedHardwareInput,
        iterations: int = 200,
        initial_temp: float = 5.0,
        seed: int = 42,
    ) -> MappingState:
        """Run gt.mcmc_anneal with a hard 10-second wall-clock time limit."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        state.init_random(seed=seed)

        objective = mapping_input.objective
        gt_state = GraphToolCompatibleState(state, objective, seed)

        beta_0 = 1.0 / initial_temp
        beta_1 = beta_0 * 1000.0

        mcmc_equilibrate_args = {
            "gibbs": False,
            "multiflip": False,
            "wait": 1,
            "force_niter": 1,
        }

        # Run mcmc_anneal in a daemon thread so we can time it out cleanly.
        exc_holder: list = []

        def _run():
            try:
                gt_mcmc.mcmc_anneal(
                    gt_state,
                    beta_range=(beta_0, beta_1),
                    niter=iterations,
                    mcmc_equilibrate_args=mcmc_equilibrate_args,
                    history=False,
                    verbose=False,
                )
            except Exception as e:
                exc_holder.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=_MCMC_TIME_LIMIT_S)

        if exc_holder:
            raise exc_holder[0]

        # Restore best mapping seen regardless of whether we timed out
        gt_state.restore_best()
        return gt_state.state
