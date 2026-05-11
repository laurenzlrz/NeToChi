import os
import subprocess
import tempfile
import time
from typing import Any
import pulp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import BaseMapper, MosaicMappingState
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.constants import MCMC_TIME_LIMIT_S

_ILP_MAX_NEURONS = 100


class ILPMapper(BaseModel, BaseMapper[MosaicMappingState[Any], MosaicMappingInput[Any]]):
    """
    Mapper formulating the problem as a Mixed Integer Linear Program (MILP).
    """
    model_config = ConfigDict(frozen=True)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S)
    max_neurons: int = Field(default=_ILP_MAX_NEURONS)

    def run(self, mapping_input: MosaicMappingInput[Any]) -> MosaicMappingState[Any]:
        """Solve for optimal mapping using the PuLP MILP solver."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        t_start = time.monotonic()

        N = graph.num_vertices()
        if N > self.max_neurons:
            raise ValueError(f"ILPMapper: problem size N={N} exceeds limit of {self.max_neurons} neurons")

        # Initialize result state
        state = MosaicMappingState.from_input(mapping_input)

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
            state.init_random()
            return state

        pair = pulp.LpVariable.dicts("pair", ((i, j, c1, c2) for (i, j) in edges for c1 in range(C) for c2 in range(C)), cat='Binary')
        
        validCross = {}
        for (i, j) in edges:
            for d in range(1, max_d + 1):
                n_slices = hw.num_slices_at_distance(d)
                for sigma in range(n_slices):
                    validCross[(i, j, d, sigma)] = pulp.LpVariable(f"vc_{i}_{j}_{d}_{sigma}", cat='Binary')

        if _budget_remaining() <= 1:
            state.init_random()
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
            state.init_random()
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

        # 4. Solve
        remaining = _budget_remaining()
        if remaining <= 1:
            state.init_random()
            return state

        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as mps_file:
            mps_path = mps_file.name
        sol_path = mps_path.replace(".mps", ".sol")

        try:
            prob.writeMPS(mps_path)
            cbc_path = pulp.PULP_CBC_CMD().path
            subprocess.run([cbc_path, mps_path, f"sec {int(remaining)}", "solve", f"solu {sol_path}"],
                           timeout=remaining + 2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(sol_path):
                prob_vars = prob.variablesDict()
                with open(sol_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3 and parts[1] in prob_vars:
                            prob_vars[parts[1]].varValue = float(parts[2])
        finally:
            for path in (mps_path, sol_path):
                if os.path.exists(path): os.unlink(path)

        # 5. Extract
        for i in range(N):
            for c in range(C):
                for x in range(X):
                    if pulp.value(w[i, c, x]) is not None and pulp.value(w[i, c, x]) > 0.5:
                        state.neuron_core_idxs_assignment[i] = c
                        state.neuron_local_idxs_assignment[i] = x
            for d in range(1, max_d + 1):
                for sigma in range(hw.num_slices_at_distance(d)):
                    if pulp.value(s_vars[(i, d, sigma)]) is not None and pulp.value(s_vars[(i, d, sigma)]) > 0.5:
                        state.neuron_slice_assignments[i, d] = sigma
        return state
