from collections import Counter
from typing import Optional

import graph_tool as gt
import icontract
import numpy as np
import pulp
from pydantic import BaseModel, Field

from netochi.input_generator.interfaces import MosaicAssignment, MosaicHardwareConfig, MosaicMappingInput
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState

DEFAULT_TIME_LIMIT_S = 500

class ILPMapperConfig(BaseModel):
    time_limit_s: float = Field(default=DEFAULT_TIME_LIMIT_S, description="Time limit for the ILP solver in seconds.")

    def create(self) -> "ILPMapper":
        return ILPMapper(config=self)


class ILPMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Exact mapper: formulates the mapping problem as an Integer Linear Program (ILP).

    Variables (all binary):
        M[i, c, x]  neuron i is placed on core c at index x
        S[i, d, s]  target neuron i listens to slice s at distance d
        I[(j, i)]   edge j -> i is inconsistent (cannot be realized)

    Objective: minimize the (multiplicity-weighted) number of inconsistent edges.
    Fan-out is not modelled; it is configured exactly after the mapping.
    """

    @icontract.require(lambda config: isinstance(config, ILPMapperConfig))
    def __init__(self, config: ILPMapperConfig) -> None:
        self.config = config
        self.time_limit_s = config.time_limit_s

    def run_mapping(
            self,
            mapping_input: MosaicMappingInput
    ) -> MosaicNetworkMappingState:

        # =========================================================================
        # 1. Structural primitives
        # =========================================================================
        graph: gt.Graph = mapping_input.graph
        hw: MosaicHardwareConfig = mapping_input.hw_config

        vertices = [int(v) for v in graph.get_vertices()]
        num_neurons = len(vertices)
        cores = list(range(hw.total_cores))
        indices = list(range(hw.neurons_per_core))

        # (source, target) pairs. Duplicate edges are merged into one variable and
        # weighted by their multiplicity in the objective, so every edge still counts.
        edge_counts = Counter((int(s), int(t)) for s, t in graph.get_edges())
        edges = list(edge_counts.keys())

        # =========================================================================
        # 2. Problem and decision variables
        # =========================================================================
        prob = pulp.LpProblem("Minimize_FanIn_Inconsistencies", pulp.LpMinimize)

        M = pulp.LpVariable.dicts(
            "M", ((i, c, x) for i in vertices for c in cores for x in indices), cat="Binary"
        )

        S_vars = {}
        for i in vertices:
            for d in range(1, hw.max_distance + 1):
                for s in range(hw.num_slices_at_distance(d)):
                    S_vars[(i, d, s)] = pulp.LpVariable(f"S_{i}_{d}_{s}", cat="Binary")

        I = pulp.LpVariable.dicts("I", edges, cat="Binary")

        # Core indicator (an expression, not a new variable): Q[i, c] = 1 iff neuron i is on core c.
        Q = {(i, c): pulp.lpSum(M[i, c, x] for x in indices) for i in vertices for c in cores}

        # =========================================================================
        # 3. Objective
        # =========================================================================
        prob += pulp.lpSum(edge_counts[e] * I[e] for e in edges), "Total_Inconsistencies"

        # =========================================================================
        # 4. Hard constraints
        # =========================================================================
        # Every neuron goes to exactly one hardware slot
        for i in vertices:
            prob += pulp.lpSum(M[i, c, x] for c in cores for x in indices) == 1

        # At most one neuron per slot (injectivity, core capacity)
        for c in cores:
            for x in indices:
                prob += pulp.lpSum(M[i, c, x] for i in vertices) <= 1

        # Every neuron listens to exactly one slice per distance
        for i in vertices:
            for d in range(1, hw.max_distance + 1):
                prob += pulp.lpSum(S_vars[(i, d, s)] for s in range(hw.num_slices_at_distance(d))) == 1

        # =========================================================================
        # 5. Slice rule (linearized AND, no big-M needed)
        # =========================================================================
        # If target i is on core c_i, source j is on core c_j at index x_j (distance d > 0),
        # and i does NOT listen to the slice containing x_j, then edge j -> i is inconsistent.
        # The target's own index does not matter, so we use the core indicator Q[i, c_i]
        # instead of looping over x_i: n times fewer constraints and a tighter LP relaxation.
        # Edges inside one core (d = 0) are always realizable and need no constraint.
        distance = {(a, b): hw.core_distance(a, b) for a in cores for b in cores}
        slice_of = {(d, x): hw.get_slice_idx(d, x)
                    for d in range(1, hw.max_distance + 1) for x in indices}

        for (j, i) in edges:  # j = source neuron, i = target neuron
            for c_i in cores:
                for c_j in cores:
                    d = distance[(c_i, c_j)]
                    if d == 0:
                        continue
                    for x_j in indices:
                        s_req = slice_of[(d, x_j)]
                        prob += I[(j, i)] >= Q[i, c_i] + M[j, c_j, x_j] - 1 - S_vars[(i, d, s_req)]

        # =========================================================================
        # 6. Solve
        # =========================================================================
        # pip's pulp bundles a CBC binary (PULP_CBC_CMD); conda-forge's pulp relies on the
        # coin-or-cbc package's `cbc` on PATH instead (COIN_CMD).
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=self.time_limit_s)
        if not solver.available():
            solver = pulp.COIN_CMD(msg=True, timeLimit=self.time_limit_s)
        prob.solve(solver)

        # After a time limit, CBC can report status "Optimal" without having proven optimality,
        # so the solution status is checked as well.
        sol_status = getattr(prob, "sol_status", None)
        proven_optimal = sol_status == getattr(pulp, "LpSolutionOptimal", 1)
        has_solution = sol_status in (
            getattr(pulp, "LpSolutionOptimal", 1),
            getattr(pulp, "LpSolutionIntegerFeasible", 2),
        ) if sol_status is not None else pulp.LpStatus[prob.status] == "Optimal"

        if not has_solution:
            raise RuntimeError(
                f"ILP solver found no feasible mapping within {self.time_limit_s} s "
                f"(status: {pulp.LpStatus[prob.status]}, solution status: {sol_status})."
            )
        if not proven_optimal:
            print(f"ILP stopped before proving optimality (status: {pulp.LpStatus[prob.status]}, "
                  f"solution status: {sol_status}). Using the best mapping found.")

        # =========================================================================
        # 7. Output extraction: binary variables -> MosaicAssignment
        # =========================================================================
        neuron_core_pre_assignment = np.full(num_neurons, -1, dtype=np.int64)
        neuron_idx_pre_assignment = np.full(num_neurons, -1, dtype=np.int64)
        neuron_slice_assignment = np.zeros((num_neurons, hw.router_levels + 1), dtype=np.int64)

        def is_one(val: Optional[float], tolerance: float = 1e-4) -> bool:
            return val is not None and abs(val - 1.0) < tolerance

        for i in vertices:
            slot = next(((c, x) for c in cores for x in indices if is_one(pulp.value(M[i, c, x]))), None)
            if slot is None:
                raise RuntimeError(f"ILP solution assigns no hardware slot to neuron {i}.")
            neuron_core_pre_assignment[i], neuron_idx_pre_assignment[i] = slot

            for d in range(1, hw.max_distance + 1):
                chosen = next((s for s in range(hw.num_slices_at_distance(d))
                               if is_one(pulp.value(S_vars[(i, d, s)]))), None)
                if chosen is None:
                    raise RuntimeError(f"ILP solution selects no slice for neuron {i} at distance {d}.")
                neuron_slice_assignment[i, d] = chosen

        # =========================================================================
        # 8. Return the mapping
        # =========================================================================
        assignment = MosaicAssignment(
            hw=hw,
            neuron_core_pre_assignment=neuron_core_pre_assignment,
            neuron_idx_pre_assignment=neuron_idx_pre_assignment,
            neuron_slice_assignment=neuron_slice_assignment,
        )
        return MosaicNetworkMappingState(
            mapping_input=mapping_input,
            assignment=assignment,
        )