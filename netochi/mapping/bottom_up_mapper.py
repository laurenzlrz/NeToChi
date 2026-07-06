import urllib
from collections import defaultdict
from typing import Optional, Dict, List

import graph_tool.all as gt
import icontract
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, Field
from scipy.optimize import milp, LinearConstraint, Bounds, OptimizeResult
import gurobipy as gp
from gurobipy import GRB
import kahypar
from collections import defaultdict

from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import KaMinParHierarchicalClusterer
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner


class BottomUpILPMapperConfig(BaseModel):
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")

    def create(self) -> "BottomUpILPMapper":
        return BottomUpILPMapper(config=self)


class BottomUpILPMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper that uses a bottom-up dual-hierarchy approach.
    Step 1: Groups neurons into Cores (randomly for this prototype).
    Step 2: Recursively splits slices globally using strictly balanced ILP Bisection.
    """

    @icontract.require(lambda config: isinstance(config, BottomUpILPMapperConfig))
    def __init__(self, config: BottomUpILPMapperConfig) -> None:
        self.config = config
        self.seed = config.seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        hw = mapping_input.hw_config
        graph = mapping_input.graph

        # ======== cluster nodes into cores and slices ===========
        temp_graph = graph
        hierarchy_mappings: List[Dict[int, int]] = []
        local_address_bit_string: np.ndarray = np.zeros(graph.num_vertices(), dtype=int)
        current_original_vertex_to_cluster: np.ndarray = np.arange(graph.num_vertices(), dtype=int)

        gt_local_idx = mapping_input.assignment.neuron_idx_pre_assignment
        total_bits = hw.router_levels

        for d in range(hw.router_levels + 1): # d = 0, 1, ..., dmax
            print(f"  [log] reached distance {d}/{hw.router_levels}")
            # Cluster temp graph
            if d == 0:
                # first level: group neurons into cores
                target_k = int(temp_graph.num_vertices() / mapping_input.hw_config.neurons_per_core)
            else:
                # subsequent levels: group clusters into subtrees
                target_k = int(temp_graph.num_vertices() / mapping_input.hw_config.nodes_per_router)
            temp_clustering = KaMinParHierarchicalClusterer()._partition_cluster(graph=temp_graph, nodes_in_cluster=None, nr_clusters=target_k)

            # Map each old temp vertex ID to its new cluster ID
            temp_vertex_to_cluster: Dict[int, int] = {}
            for cluster_id, nodes in enumerate(temp_clustering):
                for v_id in nodes:
                    temp_vertex_to_cluster[int(v_id)] = cluster_id
            hierarchy_mappings.append(temp_vertex_to_cluster)
            for global_id in range(graph.num_vertices()):
                old_temp_node = current_original_vertex_to_cluster[global_id]
                current_original_vertex_to_cluster[global_id] = temp_vertex_to_cluster[old_temp_node]

            if d == 0:
                base_core_assignment = current_original_vertex_to_cluster.copy()
            if d < hw.router_levels:
                # update the local address bit string
                local_slice_split_assignment: np.ndarray = self._split_slice_clusters(local_address_bit_string=local_address_bit_string,
                                                                                      graph=graph,
                                                                                      current_cluster_assignment=current_original_vertex_to_cluster,
                                                                                      base_core_assignment=base_core_assignment,
                                                                                      gt_local_idx=gt_local_idx,
                                                                                      current_d=d,
                                                                                      total_bits=total_bits)
                local_address_bit_string = (local_address_bit_string << 1) | local_slice_split_assignment.astype(int)

            # Overwrite temp_graph for the next hierarchy level
            temp_graph = self._build_new_temp_graph(temp_graph=temp_graph, nr_new_vertices=len(temp_clustering), vertex_to_cluster=temp_vertex_to_cluster)

        # ======== infer core assignment ============
        core_assignment: np.ndarray = self._infer_core_assignment(hierarchy_mappings=hierarchy_mappings, num_vertices=graph.num_vertices())

        # ======== infer local assignment ===========
        local_assignment = np.zeros(graph.num_vertices(), dtype=int)

        # Group all neurons by their assigned core
        core_to_neurons = defaultdict(list)
        for neuron_id in range(graph.num_vertices()):
            core_id = core_assignment[neuron_id]
            core_to_neurons[core_id].append(neuron_id)

        # Assign contiguous local indices per core based on the slice bit string
        for core_id, neurons_in_core in core_to_neurons.items():
            # Sort neurons in this core:
            # 1. Primarily by their slice ID (local_address_bit_string)
            # 2. Secondarily by their global neuron_id (to guarantee deterministic results)
            neurons_in_core.sort(key=lambda n: (local_address_bit_string[n], n))

            # Map them sequentially.
            # Example: If N=8, Slice 0 gets 0,1,2,3 and Slice 1 gets 4,5,6,7
            for local_idx, neuron_id in enumerate(neurons_in_core):
                local_assignment[neuron_id] = local_idx

        # ======== convert into mapping state =======
        clustering = ClusterAndHwOutput(cluster_assignment=core_assignment, num_clusters=mapping_input.hw_config.total_cores, hw=mapping_input.hw_config)
        slice_assignment = OptimalSliceAssigner().assign_slices(clustering, graph=graph, local_assignment=local_assignment)
        assignment = MosaicAssignment(
            hw=mapping_input.hw_config,
            neuron_core_pre_assignment=core_assignment,
            neuron_idx_pre_assignment=local_assignment,
            neuron_slice_assignment=slice_assignment
        )
        return MosaicNetworkMappingState(assignment=assignment, mapping_input=mapping_input)


    def _infer_core_assignment(self, hierarchy_mappings, num_vertices) -> np.ndarray:
        # Build top-down parent-to-child maps: children_map[level][parent_id] = [child_ids...]
        children_map = [{} for _ in range(len(hierarchy_mappings))]
        for level in range(1, len(hierarchy_mappings)):
            for child, parent in hierarchy_mappings[level].items():
                children_map[level].setdefault(parent, []).append(child)

        # Walk the tree using DFS to find the structurally ordered leaves (cores)
        ordered_cores = []
        top_level_parents = sorted(list(set(hierarchy_mappings[-1].values())))
        if len(top_level_parents) > 1:
            raise Exception("top level parents are more than one")

        def dfs(level: int, current_node: int):
            if level == 0:
                ordered_cores.append(current_node)
                return

            # Recurse into children at the level below
            for child in children_map[level].get(current_node, []):
                dfs(level - 1, child)

        # Initiate DFS from the highest level
        top_level_index = len(hierarchy_mappings) - 1
        for root_id in top_level_parents:
            dfs(top_level_index, root_id)

        # `ordered_cores` now contains KaMinPar's arbitrary d=0 cluster IDs in strict topological order.
        # Map them to contiguous sequential indices (0, 1, 2, ..., C-1).
        old_core_to_new_core = {old_id: new_id for new_id, old_id in enumerate(ordered_cores)}

        # Finally, assign each original neuron to its new, hierarchy-aware core ID.
        # We initialize an array of length N (assuming neuron IDs are 0 to N-1)
        core_assignment = np.zeros(num_vertices, dtype=int)
        for neuron_id, old_core_id in hierarchy_mappings[0].items():
            core_assignment[neuron_id] = old_core_to_new_core[old_core_id]
        return core_assignment

    def _build_new_temp_graph(self, temp_graph, nr_new_vertices, vertex_to_cluster):
        # Initialize the new graph
        new_temp_graph = gt.Graph(directed=temp_graph.is_directed())
        new_temp_graph.add_vertex(nr_new_vertices)
        # Add edges between the new clusters
        new_edges = []
        for edge in temp_graph.edges():
            src_cluster = vertex_to_cluster[int(edge.source())]
            tgt_cluster = vertex_to_cluster[int(edge.target())]
            if src_cluster != tgt_cluster:
                new_edges.append((src_cluster, tgt_cluster))
        new_temp_graph.add_edge_list(new_edges)
        return new_temp_graph

    def _split_slice_clusters(self, local_address_bit_string: np.ndarray, graph: gt.Graph, current_cluster_assignment: np.ndarray, base_core_assignment: np.ndarray, gt_local_idx: np.ndarray, current_d: int, total_bits: int) -> np.ndarray:
        n_vertices = graph.num_vertices()
        split_slice_assignment = np.zeros(n_vertices, dtype=bool)

        # 1. Identify cut edges from the latest clustering
        target_to_sources = defaultdict(list)
        # get_edges() returns an Nx2 (or Nx3) numpy array of all edges
        edges = graph.get_edges()
        sources = edges[:, 0]
        targets = edges[:, 1]
        # Vectorized lookup and comparison
        diff_mask = current_cluster_assignment[sources] != current_cluster_assignment[targets]
        # Filter down to only the edges that cross router boundaries
        cut_sources = sources[diff_mask]
        cut_targets = targets[diff_mask]

        for s, t in zip(cut_sources, cut_targets):
            target_to_sources[t].append(s)

        # 2. Iterate through each independent parent slice
        unique_parent_slices = set(local_address_bit_string)
        for parent_slice in unique_parent_slices:
            nodes_in_parent_slice = np.where(local_address_bit_string == parent_slice)[0]
            if len(nodes_in_parent_slice) == 0:
                raise Exception(f"no nodes in slice {parent_slice}")

            nodes_in_parent_slice_set = set(nodes_in_parent_slice)
            node2idx = {node: i for i, node in enumerate(nodes_in_parent_slice)}

            # 3. Build the constraint graph with WEIGHTED edges


            edge_weights = defaultdict(float)
            for target, sources in target_to_sources.items():
                valid_sources = [s for s in sources if s in nodes_in_parent_slice_set]

                # Connect ALL pairs of valid sources (fully connected clique)
                # for i in range(len(valid_sources)):
                #     for j in range(i + 1, len(valid_sources)):
                #         u = valid_sources[i]
                #         w = valid_sources[j]
                #         # Sort the tuple to ensure A->B and B->A map to the exact same edge
                #         edge = tuple(sorted((u, w)))
                #         edge_weights[edge] += 1.0

                # Connect them using a Star Graph (O(N) edges instead of O(N^2))
                if len(valid_sources) > 1:
                    hub = valid_sources[0]
                    for i in range(1, len(valid_sources)):
                        w = valid_sources[i]
                        edge = tuple(sorted((hub, w)))
                        edge_weights[edge] += 1.0

                # TODO: alternatively connect them linearly
                # for i in range(len(valid_sources) - 1):
                #     u = valid_sources[i]
                #     w = valid_sources[i + 1]
                #     # Sort the tuple to ensure A->B and B->A map to the exact same edge
                #     edge = tuple(sorted((u, w)))
                #     edge_weights[edge] += 1.0

            # Group nodes by their cluster to enforce the 50/50 split per cluster
            nodes_by_cluster = defaultdict(list) # nodes_by_cluster[i]: contains all valid nodes in core i
            for node in nodes_in_parent_slice:
                nodes_by_cluster[base_core_assignment[node]].append(node)

            # 4. Build the ILP
            n_edges = len(edge_weights)
            n_nodes = len(nodes_in_parent_slice)
            if n_edges == 0:
                print("no edges!!")
                for cluster_nodes in nodes_by_cluster.values():
                    half = len(cluster_nodes) // 2
                    for i, node in enumerate(cluster_nodes):
                        split_slice_assignment[node] = (i >= half)
                continue

            # res: OptimizeResult = self._build_ilp(nodes_by_cluster=nodes_by_cluster, edge_weights=edge_weights, node2idx=node2idx, nodes_in_slice=nodes_in_parent_slice)
            # if res.success:
            #     x_opt = np.round(res.x[:n_nodes]).astype(bool)
            #     for i, node in enumerate(nodes_in_parent_slice):
            #         split_slice_assignment[node] = x_opt[i]
            # else:
            #     raise Exception("ILP could not be solved")

            # local_opt = self._kahypar_2way_slice_split(
            #     nodes_by_cluster=nodes_by_cluster,
            #     edge_weights=edge_weights,
            #     node2idx=node2idx,
            #     nodes_in_slice=nodes_in_parent_slice
            # )
            # # --- DEBUG OBJECTIVE CHECK ---
            # kahypar_cost = 0.0
            # for (u, v), w in edge_weights.items():
            #     if local_opt[node2idx[u]] != local_opt[node2idx[v]]:
            #         kahypar_cost += w
            # gt_local_opt = np.zeros(n_nodes, dtype=bool)
            # for i, node in enumerate(nodes_in_parent_slice):
            #     gt_val = int(gt_local_idx[node])
            #     gt_bit = (gt_val >> (total_bits - current_d)) & 1
            #     gt_local_opt[i] = bool(gt_bit)
            # gt_cost = 0.0
            # for (u, v), w in edge_weights.items():
            #     if gt_local_opt[node2idx[u]] != gt_local_opt[node2idx[v]]:
            #         gt_cost += w
            # print(f"    [Obj Check] Parent {parent_slice}: KaHyPar Cut = {kahypar_cost} | Ground Truth Cut = {gt_cost}")
            # # Map the local 0/1 assignments back to the global tracker
            # for i, node in enumerate(nodes_in_parent_slice):
            #     split_slice_assignment[node] = local_opt[i]

            # solve using KL
            assignment_dict = self._kl_constrained_bisection(
                nodes_by_cluster=nodes_by_cluster,
                edge_weights=edge_weights,
                nodes_in_slice=nodes_in_parent_slice
            )

            for node, is_slice_1 in assignment_dict.items():
                split_slice_assignment[node] = is_slice_1


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

        return split_slice_assignment

    def _build_ilp(self, nodes_by_cluster, edge_weights, node2idx, nodes_in_slice) -> OptimizeResult:
        n_edges = len(edge_weights)
        n_nodes = len(nodes_in_slice)

        total_vars = n_nodes + n_edges

        # Objective: Minimize sum(W_j * z_j)
        c = np.zeros(total_vars)
        # Apply the accumulated weight to the objective coefficient of each z variable
        for j, weight in enumerate(edge_weights.values()):
            c[n_nodes + j] = weight

        num_xor_constraints = 2 * n_edges
        num_size_constraints = len(nodes_by_cluster)
        total_constraints = num_xor_constraints + num_size_constraints

        A = sp.lil_matrix((total_constraints, total_vars))
        lb = np.zeros(total_constraints)
        ub = np.full(total_constraints, np.inf)

        row = 0

        # Constraint A: Edge Cut Penalties (z_j >= |x_u - x_w|)
        for j, ((u, w), weight) in enumerate(edge_weights.items()):
            u_idx = node2idx[u]
            w_idx = node2idx[w]
            z_idx = n_nodes + j

            # z_j >= x_w - x_u
            A[row, u_idx] = 1
            A[row, w_idx] = -1
            A[row, z_idx] = 1
            lb[row] = 0
            row += 1

            # z_j >= x_u - x_w
            A[row, u_idx] = -1
            A[row, w_idx] = 1
            A[row, z_idx] = 1
            lb[row] = 0
            row += 1

        # Constraint B: Strict 50/50 Split per Cluster
        for cluster_nodes in nodes_by_cluster.values():
            target_capacity = len(cluster_nodes) // 2
            for node in cluster_nodes:
                A[row, node2idx[node]] = 1
            lb[row] = target_capacity
            ub[row] = target_capacity
            row += 1

        # Bounds and Integrality
        bounds = Bounds(np.zeros(total_vars), np.ones(total_vars))
        integrality = np.ones(total_vars)  # All boolean

        # Solve ILP
        constraints = LinearConstraint(A.tocsr(), lb, ub)
        res: OptimizeResult = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        return res


    def _build_ilp_gurobi(self, nodes_by_cluster, edge_weights, node2idx, nodes_in_slice):
        n_nodes = len(nodes_in_slice)
        n_edges = len(edge_weights)

        # 1. Initialize Model
        m = gp.Model("slice_split")
        m.setParam('OutputFlag', 0)  # Suppress console output to keep your logs clean

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

    import numpy as np
    from collections import defaultdict

    def _kl_constrained_bisection(self, nodes_by_cluster, edge_weights, nodes_in_slice, max_passes=10, num_restarts=10):
        """
        Performs a Constrained Kernighan-Lin bisection with Multi-start.
        Maintains a strict 50/50 split per cluster while minimizing the edge cut.
        """
        import numpy as np
        from collections import defaultdict

        n_nodes = len(nodes_in_slice)

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
                np.random.shuffle(cluster_nodes)
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
            nodes_in_slice: np.ndarray
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

        # 4. Partition
        kahypar.partition(hypergraph, context)

        # 5. Collect local results
        local_assignment = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            local_assignment[i] = (hypergraph.blockID(i) == 1)

        return local_assignment
