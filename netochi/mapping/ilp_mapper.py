import pulp
import numpy as np
import graph_tool as gt
from typing import Optional
from pydantic import Field, BaseModel, ConfigDict

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.definitions.constants import MCMC_TIME_LIMIT_S

from netochi.input_generator.interfaces import MosaicHardwareConfig, MosaicAssignment, MosaicMappingInput

class ILPMapper(BaseModel, BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper formulating the problem as a Mixed Integer Linear Program (MILP).
    """
    model_config = ConfigDict(frozen=True)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S, description="Time limit for the ILP solver in seconds.")

    def run(
            self,
            mapping_input: MosaicMappingInput
    ) -> MosaicNetworkMappingState:

        # 1. Dynamically extract structural primitives from your gt.Graph
        # get_vertices() returns a 1D array of vertex IDs: [0, 1, 2, ... |V|-1]
        graph: gt.Graph = mapping_input.graph
        hw: MosaicHardwareConfig = mapping_input.hw_config

        vertices = graph.get_vertices().tolist()

        # get_edges() returns a 2D array of shape (|E|, 2) representing (source, target) pairs
        edges = [tuple(edge) for edge in graph.get_edges()]

        num_neurons = len(vertices)
        cores = list(range(hw.total_cores))
        indices = list(range(hw.neurons_per_core))

        # 2. Initialize the Optimization Problem
        prob = pulp.LpProblem("Minimize_FanIn_Inconsistencies", pulp.LpMinimize)

        # 3. Decision Variables
        # M[i, c, x] = 1 if neuron i is mapped to core c at index x
        M = pulp.LpVariable.dicts("M",
                                  ((i, c, x) for i in vertices for c in cores for x in indices),
                                  cat='Binary')

        # S_vars[(i, d, s)] = 1 if neuron i selects slice s for distance d
        S_vars = {}
        for i in vertices:
            for d in range(1, hw.max_distance + 1):
                num_slices = hw.num_slices_at_distance(d)
                for s in range(num_slices):
                    S_vars[(i, d, s)] = pulp.LpVariable(f"S_{i}_{d}_{s}", cat='Binary')

        # I[j, i] = 1 if directed edge (j -> i) is inconsistent with the Fan-In masks
        I = pulp.LpVariable.dicts("I", edges, cat='Binary')

        # 4. Objective Function: Minimize total Fan-In structural penalties
        prob += pulp.lpSum(I[edge] for edge in edges), "Total_Inconsistencies"

        # 5. Hard Constraints (P)
        # Injection rule: Every neuron from the graph must go to exactly one hardware slot
        for i in vertices:
            prob += pulp.lpSum(M[i, c, x] for c in cores for x in indices) == 1

        # Capacity rule: Prevent address collisions (max one neuron per slot)
        for c in cores:
            for x in indices:
                prob += pulp.lpSum(M[i, c, x] for i in vertices) <= 1

        # Slicing rule: Every neuron configures exactly one listening slice per distance level globally [cite: 115, 117]
        for i in vertices:
            for d in range(1, hw.max_distance + 1):
                num_slices = hw.num_slices_at_distance(d)
                prob += pulp.lpSum(S_vars[(i, d, s)] for s in range(num_slices)) == 1

        # 6. Soft Constraints (Linearized Fan-In Violation Logic)
        for (j, i) in edges:  # j is source neuron, i is target neuron
            for c_i in cores:
                for x_i in indices:
                    for c_j in cores:
                        d = hw.core_distance(c_i, c_j)

                        if d > 0:
                            for x_j in indices:
                                # Figure out which slice index covers source index x_j at this distance
                                s_req = hw.get_slice_idx(d, x_j)

                                # Big-M / Penalty Linearization:
                                # If target i is at (c_i, x_i) AND source j is at (c_j, x_j)
                                # AND target i DID NOT select s_req, then I[(j, i)] must be forced to 1.
                                prob += I[(j, i)] >= (M[i, c_i, x_i] + M[j, c_j, x_j] - 1 - S_vars[(i, d, s_req)])

        # 7. Fire up the solver engine
        prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=300))

        if pulp.LpStatus[prob.status] not in ["Optimal"]:
            print(f"Solver status: {pulp.LpStatus[prob.status]}. Proceeding to extract best found state.")

        # =========================================================================
        # 8. OUTPUT EXTRACTION: Converting binary variables to MosaicAssignment
        # =========================================================================
        neuron_core_pre_assignment = np.zeros(num_neurons, dtype=np.int64)
        neuron_idx_pre_assignment = np.zeros(num_neurons, dtype=np.int64)
        neuron_slice_assignment = np.zeros((num_neurons, hw.router_levels + 1), dtype=np.int64)

        # Floating point safe checker for binary solution variants
        def cleanly_matches_one(val: Optional[float], tolerance: float = 1e-4) -> bool:
            return val is not None and abs(val - 1.0) < tolerance

        for i in vertices:
            # Extract location assignments
            for c in cores:
                found_slot = False
                for x in indices:
                    if cleanly_matches_one(pulp.value(M[i, c, x])):
                        neuron_core_pre_assignment[i] = c
                        neuron_idx_pre_assignment[i] = x
                        found_slot = True
                        break
                if found_slot:
                    break

            # Extract globally chosen fan-in slices per distance level [cite: 115, 117]
            for d in range(1, hw.max_distance + 1):
                num_slices = hw.num_slices_at_distance(d)
                for s in range(num_slices):
                    if cleanly_matches_one(pulp.value(S_vars[(i, d, s)])):
                        neuron_slice_assignment[i, d] = s
                        break

        # 9. Return the fully configured and checked object
        assignment = MosaicAssignment(
            hw=hw,
            neuron_core_pre_assignment=neuron_core_pre_assignment.as_type(np.int64),
            neuron_idx_pre_assignment=neuron_idx_pre_assignment.as_type(np.int64),
            neuron_slice_assignment=neuron_slice_assignment.as_type(np.int64)
        )
        return MosaicNetworkMappingState(
            _mapping_input=mapping_input,
            assignment=assignment
        )
