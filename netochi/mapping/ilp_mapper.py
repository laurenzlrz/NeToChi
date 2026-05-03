import os
import subprocess
import tempfile
import time

import pulp
import graph_tool.all as gt
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.core import BaseMapper, IFixedHardwareMapper, FixedHardwareInput

_ILP_TIME_LIMIT_S = 10
_ILP_MAX_NEURONS = 100  # Skip ILP for problems larger than this


class ILPMapper(BaseMapper, IFixedHardwareMapper):
    """Mapper formulating the problem as a Mixed Integer Linear Program (MILP).

    Enforces a hard wall-clock budget of _ILP_TIME_LIMIT_S seconds for the
    entire execution (model construction + solve). Automatically skips
    problems with more than _ILP_MAX_NEURONS neurons.
    """

    def map_fixed_hardware(self, mapping_input: FixedHardwareInput) -> MappingState:
        """Solve for optimal mapping using the PuLP MILP solver."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        t_start = time.monotonic()

        N = state.N
        if N > _ILP_MAX_NEURONS:
            raise NotImplementedError(
                f"ILPMapper: problem size N={N} exceeds limit of {_ILP_MAX_NEURONS} neurons"
            )

        C = mapping_input.hw_config.total_cores
        X = mapping_input.hw_config.neurons_per_core
        max_d = mapping_input.hw_config.max_distance

        # Create the model
        prob = pulp.LpProblem("NeuromorphicMapping", pulp.LpMaximize)

        # Identify all directed edges
        edges = []
        for e in mapping_input.graph.edges():
            src, tgt = int(e.source()), int(e.target())
            edges.append((src, tgt))

        # --- Budget check: abort early if model construction is too slow ---
        def _budget_remaining() -> float:
            return _ILP_TIME_LIMIT_S - (time.monotonic() - t_start)

        # 1. Variables
        # w[i, c, x]: 1 if node i is assigned to core c, address x
        w = pulp.LpVariable.dicts("w", ((i, c, x) for i in range(N) for c in range(C) for x in range(X)), cat='Binary')

        # s[i, d, sigma]: 1 if target node i selects slice sigma at distance d
        s_vars = {}
        for i in range(N):
            for d in range(1, max_d + 1):
                n_slices = mapping_input.hw_config.num_slices_at_distance(d)
                for sigma in range(n_slices):
                    s_vars[(i, d, sigma)] = pulp.LpVariable(f"s_{i}_{d}_{sigma}", cat='Binary')

        # v[i, j]: 1 if edge i -> j is valid
        v = pulp.LpVariable.dicts("v", edges, cat='Binary')

        if _budget_remaining() <= 1:
            state.init_random()
            return state

        # pair[i, j, c1, c2]: 1 if node i is in core c1 AND node j is in core c2
        pair = pulp.LpVariable.dicts("pair", ((i, j, c1, c2) for (i, j) in edges for c1 in range(C) for c2 in range(C)), cat='Binary')

        # validCross[i, j, d, sigma]: 1 if edge is valid cross-core via distance d and slice sigma
        validCross = {}
        for (i, j) in edges:
            for d in range(1, max_d + 1):
                n_slices = mapping_input.hw_config.num_slices_at_distance(d)
                for sigma in range(n_slices):
                    validCross[(i, j, d, sigma)] = pulp.LpVariable(f"vc_{i}_{j}_{d}_{sigma}", cat='Binary')

        if _budget_remaining() <= 1:
            state.init_random()
            return state

        # 2. Objective: Maximize total valid edges
        prob += pulp.lpSum(v[e] for e in edges), "MaximizeValidEdges"

        # 3. Constraints

        # A. Unique Placement: Each node in exactly one slot
        for i in range(N):
            prob += pulp.lpSum(w[i, c, x] for c in range(C) for x in range(X)) == 1

        # B. Slot Capacity: Each slot holds at most one node
        for c in range(C):
            for x in range(X):
                prob += pulp.lpSum(w[i, c, x] for i in range(N)) <= 1

        # C. Unique Slice: Each node selects exactly one slice per distance
        for i in range(N):
            for d in range(1, max_d + 1):
                n_slices = mapping_input.hw_config.num_slices_at_distance(d)
                prob += pulp.lpSum(s_vars[(i, d, sigma)] for sigma in range(n_slices)) == 1

        if _budget_remaining() <= 1:
            state.init_random()
            return state

        # D. Pairwise core tracking
        for (i, j) in edges:
            for c1 in range(C):
                for c2 in range(C):
                    i_in_c1 = pulp.lpSum(w[i, c1, x] for x in range(X))
                    j_in_c2 = pulp.lpSum(w[j, c2, x] for x in range(X))
                    prob += pair[i, j, c1, c2] <= i_in_c1
                    prob += pair[i, j, c1, c2] <= j_in_c2

        if _budget_remaining() <= 1:
            state.init_random()
            return state

        # E. Validity bounds
        for (i, j) in edges:
            same_core_expr = pulp.lpSum(pair[i, j, c, c] for c in range(C))

            cross_core_expr = []
            for d in range(1, max_d + 1):
                n_slices = mapping_input.hw_config.num_slices_at_distance(d)
                pairs_at_dist = [pair[i, j, c1, c2] for c1 in range(C) for c2 in range(C)
                                 if mapping_input.hw_config.core_distance(c2, c1) == d]
                dist_match_expr = pulp.lpSum(pairs_at_dist)

                for sigma in range(n_slices):
                    vc = validCross[(i, j, d, sigma)]
                    prob += vc <= s_vars[(j, d, sigma)]
                    prob += vc <= dist_match_expr
                    start_x, end_x = mapping_input.hw_config.get_slice_bounds(d, sigma)
                    i_in_slice_bounds = pulp.lpSum(w[i, c, x] for c in range(C) for x in range(start_x, end_x))
                    prob += vc <= i_in_slice_bounds
                    cross_core_expr.append(vc)

            prob += v[(i, j)] <= same_core_expr + pulp.lpSum(cross_core_expr)

        # 4. Solve with remaining time budget via subprocess
        remaining = _budget_remaining()
        if remaining <= 1:
            state.init_random()
            return state

        cbc_seconds = max(1, int(remaining))

        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as mps_file:
            mps_path = mps_file.name
        sol_path = mps_path.replace(".mps", ".sol")

        try:
            prob.writeMPS(mps_path)
            cbc_path = pulp.PULP_CBC_CMD().path
            cmd = [cbc_path, mps_path, f"sec {cbc_seconds}", "solve", f"solu {sol_path}"]

            try:
                subprocess.run(
                    cmd,
                    timeout=remaining + 2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                pass  # CBC killed, read whatever it wrote

            # 5. Parse the CBC solution file back into PuLP variables
            if os.path.exists(sol_path):
                prob_vars = prob.variablesDict()
                try:
                    with open(sol_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                var_name = parts[1]
                                var_val = float(parts[2])
                                if var_name in prob_vars:
                                    prob_vars[var_name].varValue = var_val
                except Exception:
                    pass
        finally:
            for path in (mps_path, sol_path):
                if os.path.exists(path):
                    os.unlink(path)

        # 6. Extract mapping back to state
        for i in range(N):
            for c in range(C):
                for x in range(X):
                    val = pulp.value(w[i, c, x])
                    if val is not None and val > 0.5:
                        state.c[i] = c
                        state.x[i] = x

            for d in range(1, max_d + 1):
                n_slices = mapping_input.hw_config.num_slices_at_distance(d)
                for sigma in range(n_slices):
                    val = pulp.value(s_vars[(i, d, sigma)])
                    if val is not None and val > 0.5:
                        state.s[i, d] = sigma

        return state
