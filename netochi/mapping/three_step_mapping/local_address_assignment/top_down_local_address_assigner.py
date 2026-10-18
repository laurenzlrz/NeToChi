from collections import defaultdict

from gurobipy import GRB
from pydantic import Field
from enum import Enum
from typing import Dict

import graph_tool.all as gt
import icontract
import numpy as np
import gurobipy as gp
import numpy.typing as npt
from pydantic import BaseModel

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput


class SliceSolveMode(Enum):
    KL_CONSTRAINED_BISECTION = 1
    KA_HY_PAR = 2

class ConstraintGraphMode(Enum):
    STAR = 1
    CLIQUE = 2
    CHAIN = 3


class TopDownLocalAddressAssignerConfig(BaseModel):
    slice_solve_mode: SliceSolveMode = Field(
        default=SliceSolveMode.KL_CONSTRAINED_BISECTION,
        description="The algorithm used to solve the slice partitioning problem",
    )
    constraint_graph_mode: ConstraintGraphMode = Field(
        default=ConstraintGraphMode.CLIQUE,
        description="The strategy used to connect common sources in constraint graph",
    )

    def create(self) -> "TopDownLocalAddressAssigner":
        return TopDownLocalAddressAssigner(config=self)


class TopDownLocalAddressAssigner(LocalAddressAssigner):
    """
    Assigns local addresses using ground truth if available.
    Otherwise, falls back to a random permutation of local indices within each cluster core.
    """

    @icontract.require(lambda config: isinstance(config, TopDownLocalAddressAssignerConfig))
    def __init__(self, config: TopDownLocalAddressAssignerConfig) -> None:
        self.config = config

    def assign_addresses(self, mapping_input: MosaicMappingInput, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:

        hw = mapping_input.hw_config
        graph = mapping_input.graph
        core_assignment = clustering.cluster_assignment

        # ======== infer local assignment bit string ===========
        local_address_bit_string: np.ndarray = np.zeros(graph.num_vertices(), dtype=int)
        for d in range(hw.router_levels):
            original_vertex_to_current_cluster: np.ndarray = clustering.cluster_assignment // (2**d)
            temp_slice_split_assignment: np.ndarray = self.split_slice_clusters(
                local_address_bit_string=local_address_bit_string,
                graph=graph,
                current_cluster_assignment=original_vertex_to_current_cluster,
                base_core_assignment=core_assignment,
                seed=mapping_input.seed
            )
            local_address_bit_string = (local_address_bit_string << 1) | temp_slice_split_assignment.astype(int)

        # ======== infer local assignment ===========
        local_assignment = np.zeros(graph.num_vertices(), dtype=int)

        # Group all neurons by their assigned core
        core_to_neurons = defaultdict(list)
        for neuron_id in range(graph.num_vertices()):
            core_id = core_assignment[neuron_id]
            core_to_neurons[core_id].append(neuron_id)

        # Assign contiguous local indices per core based on the slice bit string: ensures that every neuron receives unique local index within core even when 2^dmax < N
        for core_id, neurons_in_core in core_to_neurons.items():
            # Sort neurons in this core: Primarily by their slice ID, Secondarily by their neuron_id (for deterministic results)
            neurons_in_core.sort(key=lambda n: (local_address_bit_string[n], n))

            # Map them sequentially (Example: If N=8, Slice 0 gets 0,1,2,3 and Slice 1 gets 4,5,6,7)
            for local_idx, neuron_id in enumerate(neurons_in_core):
                local_assignment[neuron_id] = local_idx

        return local_assignment


    def split_slice_clusters(self, local_address_bit_string: np.ndarray, graph: gt.Graph, current_cluster_assignment: np.ndarray, base_core_assignment: np.ndarray, seed) -> np.ndarray:
        n_vertices = graph.num_vertices()
        split_slice_assignment = np.zeros(n_vertices, dtype=bool)

        # 1. Identify cut edges from the latest clustering: as target_to_sources
        target_to_sources = defaultdict(list)
        edges = graph.get_edges()
        sources = edges[:, 0]
        targets = edges[:, 1]
        # Filter down to only the edges that cross router boundaries
        diff_mask = current_cluster_assignment[sources] != current_cluster_assignment[targets]
        cut_sources = sources[diff_mask]
        cut_targets = targets[diff_mask]
        for s, t in zip(cut_sources, cut_targets):
            target_to_sources[t].append(s)

        # 2. Iterate through each parent slice
        unique_parent_slices = set(local_address_bit_string)
        for parent_slice in unique_parent_slices:
            nodes_in_parent_slice = np.where(local_address_bit_string == parent_slice)[0]
            if len(nodes_in_parent_slice) == 0:
                raise Exception(f"no nodes in slice {parent_slice}")

            # 3. Build the constraint graph with WEIGHTED edges: Connect valid sources with common targets
            edge_weights: Dict[int, float] = self._construct_edge_weights_of_constraint_graph(target_to_sources, nodes_in_parent_slice)

            # 4. Group nodes by their cluster to enforce the 50/50 split per cluster
            nodes_by_cluster = defaultdict(list) # nodes_by_cluster[i]: contains all valid nodes in core i
            for node in nodes_in_parent_slice:
                nodes_by_cluster[base_core_assignment[node]].append(node)

            if len(edge_weights) == 0:
                # fallback: random assignment
                print("no edges!!")
                for cluster_nodes in nodes_by_cluster.values():
                    half = len(cluster_nodes) // 2
                    for i, node in enumerate(cluster_nodes):
                        split_slice_assignment[node] = (i >= half)
                continue

            # 5. split slices using constraint graph and split constraints
            assignment_dict: Dict[int, bool]= self._solve_slice_assignment(nodes_by_cluster=nodes_by_cluster, edge_weights=edge_weights, nodes_in_parent_slice=nodes_in_parent_slice, seed=seed)
            for node, is_slice_1 in assignment_dict.items():
                split_slice_assignment[node] = is_slice_1

        return split_slice_assignment

    def _construct_edge_weights_of_constraint_graph(self, target_to_sources, nodes_in_parent_slice) -> Dict[int, float]:
        mode = self.config.constraint_graph_mode
        edge_weights = defaultdict(float)
        for target, sources in target_to_sources.items():
            valid_sources = [s for s in sources if s in set(nodes_in_parent_slice)]
            if len(valid_sources) <= 1:
                continue

            if mode == ConstraintGraphMode.STAR:
                # Connect valid sources using a Star Graph (O(N) edges instead of O(N^2))
                hub = valid_sources[0]
                for i in range(1, len(valid_sources)):
                    w = valid_sources[i]
                    edge = tuple(sorted((hub, w)))
                    edge_weights[edge] += 1.0

            if mode == ConstraintGraphMode.CLIQUE:
                # Connect valid sources as fully connected clique
                for i in range(len(valid_sources)):
                    for j in range(i + 1, len(valid_sources)):
                        u = valid_sources[i]
                        w = valid_sources[j]
                        # Sort the tuple to ensure A->B and B->A map to the exact same edge
                        edge = tuple(sorted((u, w)))
                        edge_weights[edge] += 1.0

            if mode == ConstraintGraphMode.CHAIN:
                # Connect valid sources as linear chain
                for i in range(len(valid_sources) - 1):
                    u = valid_sources[i]
                    w = valid_sources[i + 1]
                    # Sort the tuple to ensure A->B and B->A map to the exact same edge
                    edge = tuple(sorted((u, w)))
                    edge_weights[edge] += 1.0

        return edge_weights


    def _solve_slice_assignment(self, nodes_by_cluster, edge_weights, nodes_in_parent_slice, seed) -> Dict[int, bool]:
        mode = self.config.slice_solve_mode
        assignment_dict: Dict[int, bool]
        # solve using KL
        if mode == SliceSolveMode.KL_CONSTRAINED_BISECTION:
            assignment_dict = self._kl_constrained_bisection(
                nodes_by_cluster=nodes_by_cluster,
                edge_weights=edge_weights,
                nodes_in_slice=nodes_in_parent_slice,
                seed=seed
            )
        else:
            raise Exception(f"Mode {mode} not implemented yet")
        # if mode == SliceSolveMode.KA_HY_PAR:
        #     # solve using KaHyPar
        #     local_opt = self._kahypar_2way_slice_split(
        #         nodes_by_cluster=nodes_by_cluster,
        #         edge_weights=edge_weights,
        #         node2idx=node2idx,
        #         nodes_in_slice=nodes_in_parent_slice
        #     )
        #     # --- DEBUG OBJECTIVE CHECK ---
        #     kahypar_cost = 0.0
        #     for (u, v), w in edge_weights.items():
        #         if local_opt[node2idx[u]] != local_opt[node2idx[v]]:
        #             kahypar_cost += w
        #     gt_local_opt = np.zeros(n_nodes, dtype=bool)
        #     for i, node in enumerate(nodes_in_parent_slice):
        #         gt_val = int(gt_local_idx[node])
        #         gt_bit = (gt_val >> (total_bits - current_d)) & 1
        #         gt_local_opt[i] = bool(gt_bit)
        #     gt_cost = 0.0
        #     for (u, v), w in edge_weights.items():
        #         if gt_local_opt[node2idx[u]] != gt_local_opt[node2idx[v]]:
        #             gt_cost += w
        #     print(f"    [Obj Check] Parent {parent_slice}: KaHyPar Cut = {kahypar_cost} | Ground Truth Cut = {gt_cost}")
        #     # Map the local 0/1 assignments back to the global tracker
        #     for i, node in enumerate(nodes_in_parent_slice):
        #         split_slice_assignment[node] = local_opt[i]

        #if mode == ILP:
            # # Solve ILP using Gurobi
            # success, x_opt = self._build_ilp_gurobi(
            #     nodes_by_cluster=nodes_by_cluster,
            #     edge_weights=edge_weights,
            #     node2idx=node2idx,
            #     nodes_in_slice=nodes_in_parent_slice
            # )
            # if success:
            #     for i, node in enumerate(nodes_in_parent_slice):
            #         split_slice_assignment[node] = x_opt[i]
            # else:
            #     raise Exception("Gurobi ILP could not be solved or was infeasible.")

        return assignment_dict

    def _build_ilp_gurobi(self, nodes_by_cluster, edge_weights, node2idx, nodes_in_slice, seed):
        n_nodes = len(nodes_in_slice)
        n_edges = len(edge_weights)

        # 1. Initialize Model
        m = gp.Model("slice_split")
        m.setParam('OutputFlag', 0)  # Suppress console output to keep your logs clean
        m.setParam('Seed', seed)  # <--- CRITICAL: Forces Gurobi to use your pipeline seed

        # 2. Add Variables
        # x variables: binary assignment for each node (0 or 1)
        x = m.addVars(n_nodes, vtype=GRB.BINARY, name="x")

        # z variables: continuous penalty variable for each edge cut
        # (Continuous is faster to solve, and since x is binary, z will naturally resolve to 0 or 1)
        z = m.addVars(n_edges, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")

        # 3. Set Objective: Minimize sum of (weight * z_j)
        # gp.quicksum is Gurobi's highly optimized version of Python's sum()
        objective = gp.quicksum(weight * z[j] for j, weight in enumerate(edge_weights.values()))
        m.setObjective(objective, GRB.MINIMIZE)

        # 4. Add Constraints A: Edge Cut Penalties
        # z_j >= |x_u - x_w|
        for j, ((u, w), weight) in enumerate(edge_weights.items()):
            u_idx = node2idx[u]
            w_idx = node2idx[w]

            m.addConstr(z[j] >= x[u_idx] - x[w_idx], name=f"cut_pos_{j}")
            m.addConstr(z[j] >= x[w_idx] - x[u_idx], name=f"cut_neg_{j}")

        # 5. Add Constraints B: Strict 50/50 Split per Cluster
        for cluster_id, cluster_nodes in nodes_by_cluster.items():
            target_capacity = len(cluster_nodes) // 2

            m.addConstr(
                gp.quicksum(x[node2idx[node]] for node in cluster_nodes) == target_capacity,
                name=f"capacity_{cluster_id}"
            )

        # 6. Optimize the model
        m.optimize()

        # 7. Extract Results
        if m.Status == GRB.OPTIMAL:
            # Retrieve the boolean assignments from the optimized x variables
            # .X accesses the solved value. We use > 0.5 to safely handle floating point precision
            x_opt = np.array([x[i].X > 0.5 for i in range(n_nodes)], dtype=bool)
            return True, x_opt
        else:
            # Handle infeasibility or timeouts
            return False, None

    def _kl_constrained_bisection(self, nodes_by_cluster, edge_weights, nodes_in_slice, seed, max_passes=10, num_restarts=10):
        """
        Performs a Constrained Kernighan-Lin bisection with Multi-start.
        Maintains a strict 50/50 split per cluster while minimizing the edge cut.
        """
        import numpy as np
        from collections import defaultdict

        n_nodes = len(nodes_in_slice)
        rng = np.random.default_rng(seed)

        # 1. Build adjacency list for fast neighbor lookups
        adj = defaultdict(lambda: defaultdict(float))
        for (u, v), weight in edge_weights.items():
            adj[u][v] += weight
            adj[v][u] += weight

        best_global_cut = float('inf')
        best_global_assignment = {}

        # --- MULTI-START LOOP ---
        for restart in range(num_restarts):

            # 2. Initialize a valid random 50/50 assignment
            assignment = {}
            for cluster_nodes in nodes_by_cluster.values():
                rng.shuffle(cluster_nodes)
                half = len(cluster_nodes) // 2
                for i, node in enumerate(cluster_nodes):
                    assignment[node] = (i >= half)

            # 3. Helper function to calculate initial node gains
            def calculate_gains():
                gains = {}
                for node in nodes_in_slice:
                    my_slice = assignment[node]
                    ext_cost = 0.0
                    int_cost = 0.0
                    for neighbor, weight in adj[node].items():
                        if assignment[neighbor] == my_slice:
                            int_cost += weight
                        else:
                            ext_cost += weight
                    gains[node] = ext_cost - int_cost
                return gains

            gains = calculate_gains()

            # 4. Main KL Loop
            for pass_idx in range(max_passes):
                unlocked = set(nodes_in_slice)
                history = []
                cumulative_gain = 0.0
                best_cumulative_gain = 0.0
                best_step = -1

                while True:
                    best_swap_gain = -float('inf')
                    best_pair = None

                    # Find the best valid intra-core swap
                    for cluster_nodes in nodes_by_cluster.values():
                        slice_0_nodes = [n for n in cluster_nodes if n in unlocked and not assignment[n]]
                        slice_1_nodes = [n for n in cluster_nodes if n in unlocked and assignment[n]]

                        if not slice_0_nodes or not slice_1_nodes:
                            continue

                        for u in slice_0_nodes:
                            for v in slice_1_nodes:
                                edge_w = adj[u].get(v, 0.0)
                                swap_gain = gains[u] + gains[v] - (2 * edge_w)

                                if swap_gain > best_swap_gain:
                                    best_swap_gain = swap_gain
                                    best_pair = (u, v)

                    if best_pair is None:
                        break

                    u, v = best_pair
                    history.append((u, v, best_swap_gain))
                    cumulative_gain += best_swap_gain

                    if cumulative_gain > best_cumulative_gain:
                        best_cumulative_gain = cumulative_gain
                        best_step = len(history) - 1

                    # Execute swap temporarily
                    assignment[u] = not assignment[u]
                    assignment[v] = not assignment[v]
                    unlocked.remove(u)
                    unlocked.remove(v)

                    # --- THE FIX: Correct Dynamic Gain Updates ---
                    for node in [u, v]:
                        for neighbor, weight in adj[node].items():
                            if neighbor in unlocked:
                                if assignment[node] == assignment[neighbor]:
                                    # They are now in the SAME slice (Edge became Internal)
                                    # Gain (Ext - Int) must DECREASE
                                    gains[neighbor] -= 2 * weight
                                else:
                                    # They are now in DIFFERENT slices (Edge became External)
                                    # Gain (Ext - Int) must INCREASE
                                    gains[neighbor] += 2 * weight

                                    # 5. Rollback or Terminate Pass
                if best_cumulative_gain > 0:
                    for i in range(len(history) - 1, best_step, -1):
                        u, v, _ = history[i]
                        assignment[u] = not assignment[u]
                        assignment[v] = not assignment[v]
                    gains = calculate_gains()
                else:
                    for u, v, _ in reversed(history):
                        assignment[u] = not assignment[u]
                        assignment[v] = not assignment[v]
                    break

                    # 6. Evaluate Final Cut for this Restart
            current_cut = 0.0
            for u in nodes_in_slice:
                for v, weight in adj[u].items():
                    if u < v and assignment[u] != assignment[v]:
                        current_cut += weight

            # Save if this random start yielded the best result
            if current_cut < best_global_cut:
                best_global_cut = current_cut
                best_global_assignment = assignment.copy()

        return best_global_assignment

    def _kahypar_2way_slice_split(
            self,
            nodes_by_cluster: dict,
            edge_weights: dict,
            node2idx: dict,
            nodes_in_slice: np.ndarray,
            seed: int
    ) -> np.ndarray:
        """
        Splits a single parent slice into two perfectly balanced halves using KaHyPar.
        """
        import kahypar

        n_nodes = len(nodes_in_slice)
        n_edges = len(edge_weights)

        # If there are no active cut edges for this slice at this distance,
        # trivially split the nodes in each core 50/50.
        if n_edges == 0:
            local_assignment = np.zeros(n_nodes, dtype=bool)
            for cluster_nodes in nodes_by_cluster.values():
                half = len(cluster_nodes) // 2
                for i, node in enumerate(cluster_nodes):
                    local_assignment[node2idx[node]] = (i >= half)
            return local_assignment

        # 1. Translate hyperedges to local 0-indexed space for KaHyPar
        hyperedge_indices = [0]
        hyperedges_flat = []

        for (u, v), weight in edge_weights.items():
            # Using your star-graph/clique edges as 2-node hyperedges
            hyperedges_flat.extend([node2idx[u], node2idx[v]])
            hyperedge_indices.append(len(hyperedges_flat))

        node_weights = [1] * n_nodes
        edge_weights_list = [int(w) for w in edge_weights.values()]

        # 2. Create KaHyPar Hypergraph
        hypergraph = kahypar.Hypergraph(
            n_nodes,
            n_edges,
            hyperedge_indices,
            hyperedges_flat,
            2,  # K = 2 (Bisection)
            edge_weights_list,
            node_weights
        )

        # 3. Setup Context
        context = kahypar.Context()
        # Ensure you have a 'cut_rBipartition.ini' or similar bisection config file in your workspace
        context.loadINIconfiguration("/Users/lauraerb/PycharmProjects/NeToChi/cut_rKaHyPar_sea20.ini")
        context.setK(2)
        context.setEpsilon(0.0)  # Enforce strict 50/50 global balance within this slice
        context.suppressOutput(True)
        context.setSeed(seed)  # <--- CRITICAL: Seeds KaHyPar's internal C++ engine

        # 4. Partition
        kahypar.partition(hypergraph, context)

        # 5. Collect local results
        local_assignment = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            local_assignment[i] = (hypergraph.blockID(i) == 1)

        return local_assignment


    def get_name(self) -> str:
        return self.config.slice_solve_mode.name