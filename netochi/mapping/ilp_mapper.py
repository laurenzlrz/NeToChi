import time
from typing import Any
import pulp  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicHWMappingInput
from netochi.mapping.constants import MCMC_TIME_LIMIT_S

import shutil  # Make sure this is imported at the top of your file

_ILP_MAX_NEURONS = 100

# todo: only fixed with Gemini, needs to be verified

class ILPMapper(BaseModel, BaseMapper[MosaicNetworkMappingState[Any], MosaicHWMappingInput[Any]]):
    """
    Mapper formulating the problem as a Mixed Integer Linear Program (MILP).
    """
    model_config = ConfigDict(frozen=True)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S)
    max_neurons: int = Field(default=_ILP_MAX_NEURONS)

    def run(self, mapping_input: MosaicHWMappingInput[Any]) -> MosaicNetworkMappingState[Any]:
        """Solve for optimal mapping using the PuLP MILP solver."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config_inferred
        t_start = time.monotonic()

        N = graph.num_vertices()
        if N > self.max_neurons:
            raise ValueError(f"ILPMapper: problem size N={N} exceeds limit of {self.max_neurons} neurons")

        # Initialize result state
        state = MosaicNetworkMappingState.from_input(mapping_input)

        C = hw.total_cores
        X = hw.neurons_per_core
        max_d = hw.max_distance

        def _budget_remaining() -> float:
            return self.time_limit_s - (time.monotonic() - t_start)

        # 1. Variables
        w = pulp.LpVariable.dicts("w", ((i, c, x) for i in range(N) for c in range(C) for x in range(X)), cat='Binary')
        
        s_vars = {}
        for i in range(N):
            for d in range(1, max_d + 1):
                n_slices = hw.num_slices_at_distance(d)
                for sigma in range(n_slices):
                    s_vars[(i, d, sigma)] = pulp.LpVariable(f"s_{i}_{d}_{sigma}", cat='Binary')

        edges = [(int(e.source()), int(e.target())) for e in graph.edges()]
        v = pulp.LpVariable.dicts("v", edges, cat='Binary')

        if _budget_remaining() <= 1:
            state.init_random_assignments()
            return state

        pair = pulp.LpVariable.dicts("pair", ((i, j, c1, c2) for (i, j) in edges for c1 in range(C) for c2 in range(C)), cat='Binary')
        
        validCross = {}
        for (i, j) in edges:
            for d in range(1, max_d + 1):
                n_slices = hw.num_slices_at_distance(d)
                for sigma in range(n_slices):
                    validCross[(i, j, d, sigma)] = pulp.LpVariable(f"vc_{i}_{j}_{d}_{sigma}", cat='Binary')

        if _budget_remaining() <= 1:
            state.init_random_assignments()
            return state

        # 2. Objective
        prob = pulp.LpProblem("NeuromorphicMapping", pulp.LpMaximize)
        prob += pulp.lpSum(v[e] for e in edges), "MaximizeValidEdges"

        # 3. Constraints
        for i in range(N):
            prob += pulp.lpSum(w[i, c, x] for c in range(C) for x in range(X)) == 1

        for c in range(C):
            for x in range(X):
                prob += pulp.lpSum(w[i, c, x] for i in range(N)) <= 1

        for i in range(N):
            for d in range(1, max_d + 1):
                n_slices = hw.num_slices_at_distance(d)
                prob += pulp.lpSum(s_vars[(i, d, sigma)] for sigma in range(n_slices)) == 1

        if _budget_remaining() <= 1:
            state.init_random_assignments()
            return state

        for (i, j) in edges:
            for c1 in range(C):
                for c2 in range(C):
                    i_in_c1 = pulp.lpSum(w[i, c1, x] for x in range(X))
                    j_in_c2 = pulp.lpSum(w[j, c2, x] for x in range(X))
                    prob += pair[i, j, c1, c2] <= i_in_c1
                    prob += pair[i, j, c1, c2] <= j_in_c2

        for (i, j) in edges:
            same_core_expr = pulp.lpSum(pair[i, j, c, c] for c in range(C))
            cross_core_expr = []
            for d in range(1, max_d + 1):
                n_slices = hw.num_slices_at_distance(d)
                pairs_at_dist = [pair[i, j, c1, c2] for c1 in range(C) for c2 in range(C) if hw.core_distance(c2, c1) == d]
                dist_match_expr = pulp.lpSum(pairs_at_dist)
                for sigma in range(n_slices):
                    vc = validCross[(i, j, d, sigma)]
                    prob += vc <= s_vars[(j, d, sigma)]
                    prob += vc <= dist_match_expr
                    start_x, end_x = hw.get_slice_bounds(d, sigma)
                    i_in_slice_bounds = pulp.lpSum(w[i, c, x] for c in range(C) for x in range(start_x, end_x))
                    prob += vc <= i_in_slice_bounds
                    cross_core_expr.append(vc)
            prob += v[(i, j)] <= same_core_expr + pulp.lpSum(cross_core_expr)

        # 4. Solve using PuLP's native wrapper
        remaining = _budget_remaining()
        if remaining <= 1:
            state.init_random_assignments()
            return state

        # 1. Find the absolute path to the cbc binary in your Conda env
        cbc_path = shutil.which("cbc")

        if cbc_path is None:
            raise RuntimeError(
                "CRITICAL: The CBC solver is not installed or not in your PATH. "
                "Please open your terminal, activate your conda environment, and run: "
                "conda install -c conda-forge coincbc"
            )

        # 2. Use COIN_CMD (the correct class for Conda/External installations)
        solver = pulp.COIN_CMD(path=cbc_path, timeLimit=int(remaining), msg=False)

        # 3. Solve
        try:
            status = prob.solve(solver)
        except pulp.apis.core.PulpSolverError as e:
            raise RuntimeError(
                f"COIN_CMD failed to execute the binary found at {cbc_path}. "
                f"Verify permissions by running: chmod +x {cbc_path}"
            ) from e

        # If the solver failed to find ANY feasible solution, fallback to random
        if status == pulp.LpStatusInfeasible or status == pulp.LpStatusNotSolved:
            state.init_random_assignments()
            return state

        # 5. Extract results from variables into the state object
        for i in range(N):
            for c in range(C):
                for x in range(X):
                    val = pulp.value(w[i, c, x])
                    if val is not None and val > 0.5:
                        state.neuron_core_idxs_assignment[i] = c
                        state.neuron_local_idxs_assignment[i] = x

            for d in range(1, max_d + 1):
                for sigma in range(hw.num_slices_at_distance(d)):
                    val = pulp.value(s_vars[(i, d, sigma)])
                    if val is not None and val > 0.5:
                        state.neuron_slice_assignments[i, d] = sigma

        return state