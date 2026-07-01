import math
from typing import Tuple

import graph_tool.all as gt
import numpy as np
import numpy.typing as npt
import cvxpy as cp
import scipy.linalg

from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput


class SDPRoundingAssigner(LocalAddressAssigner):
    """
    Weist Neuronen lokale Adressen innerhalb eines Cores zu (0.878-Approximation).
    Das Problem wird als rekursive Bisektion eines bipartiten Sternengraphen formuliert,
    über eine SDP-Relaxierung (Goemans-Williamson) mit anschließendem Random Hyperplane
    Rounding und deterministischer Frieze-Jerrum-Projektion gelöst.
    """

    def __init__(self, time_limit_secs: float = 4.0) -> None:
        self.time_limit_secs = time_limit_secs

    def assign_addresses(self, graph: gt.Graph, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:
        """
        The top-level function that initiates the assignment and extracts just the local addresses.
        """
        local_addresses, _ = self.assign_addresses_and_slices(graph, clustering)
        return local_addresses

    def assign_addresses_and_slices(self, graph: gt.Graph, clustering: 'ClusterAndHwOutput') -> Tuple[
        npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        The intermediate function that sets up the global structures and invokes the recursion.
        Returns the local addresses and the slice assignments across router levels.
        """
        assert clustering.hw.slice_factor == 2, "SDPRoundingAssigner currently only supports slice_factor=2."
        hw = clustering.hw
        num_neurons = clustering.cluster_assignment.shape[0]

        # Initialisierung der Rückgabe-Arrays
        local_addresses = np.full(num_neurons, -1, dtype=np.int_)
        slice_assignment = np.zeros((num_neurons, hw.router_levels + 1), dtype=np.int_)

        active_neurons = np.arange(num_neurons)
        edges = graph.get_edges()
        active_edges = [(u, p) for u, p in edges]

        # Start der Rekursion
        self.recursive_assign_addresses_and_slices(
            clustering=clustering,
            active_neurons=active_neurons,
            active_edges=active_edges,
            depth=1,
            local_addr_start=0,
            local_addr_end=hw.neurons_per_core,
            local_addresses=local_addresses,
            slice_assignment=slice_assignment
        )

        # Globale Auflösung der Slice-Indizes
        for d in range(1, hw.router_levels + 1):
            slice_assignment[:, d] = slice_assignment[:, d - 1] * hw.slice_factor + slice_assignment[:, d]

        return local_addresses, slice_assignment

    def recursive_assign_addresses_and_slices(
            self,
            clustering: 'ClusterAndHwOutput',
            active_neurons: npt.NDArray[np.int_] | list[int],
            active_edges: list[tuple[int, int]],
            depth: int,
            local_addr_start: int,
            local_addr_end: int,
            local_addresses: npt.NDArray[np.int_],
            slice_assignment: npt.NDArray[np.int_]
    ) -> None:
        """
        The recursive bisection core. Replaces the LP and Pipage Rounding with
        Semidefinite Programming (SDP), random hyperplane projection, and greedy adjustments.
        """
        hw = clustering.hw

        if len(active_neurons) == 0:
            return

        if depth > hw.router_levels or local_addr_start >= local_addr_end - 1:
            cores_to_neurons = {}
            for u in active_neurons:
                c = clustering.cluster_assignment[u]
                cores_to_neurons.setdefault(c, []).append(u)

            for c, neurons in cores_to_neurons.items():
                addr = local_addr_start
                for u in neurons:
                    local_addresses[u] = addr
                    addr += 1
            return

        # Filter edges relevant at this depth
        filtered_edges = []
        for u, p in active_edges:
            core_u = clustering.cluster_assignment[u]
            core_p = clustering.cluster_assignment[p]
            if hw.core_distance(core_u, core_p) >= depth:
                filtered_edges.append((u, p))

        num_neurons = len(active_neurons)
        neuron_to_idx = {u: i for i, u in enumerate(active_neurons)}

        unique_proxies = list(set(p for u, p in filtered_edges))
        num_proxies = len(unique_proxies)
        proxy_to_idx = {p: i for i, p in enumerate(unique_proxies)}

        num_edges = len(filtered_edges)
        print(f"[SDP Assigner] Depth {depth}: Active Neurons = {num_neurons}, Filtered Edges = {num_edges}", flush=True)
        L = math.ceil((local_addr_end - local_addr_start) / 2.0)
        R = (local_addr_end - local_addr_start) - L
        C_max = max(L, R)

        cores_to_neurons_idx = {}
        for i, u in enumerate(active_neurons):
            c = clustering.cluster_assignment[u]
            cores_to_neurons_idx.setdefault(c, []).append(i)

        if num_edges == 0:
            y = np.zeros(num_neurons)
            for c, neurons_in_core_idx in cores_to_neurons_idx.items():
                for i, u_idx in enumerate(neurons_in_core_idx):
                    if i >= L: y[u_idx] = 1.0
            z_rounded = np.zeros(0, dtype=np.int_)
        else:
            # =================================================================
            # Phase 1 & 2: The Vector Encoding & Semidefinite Program (SDP)
            # =================================================================
            total_vars = num_neurons + num_proxies

            # X is our positive semidefinite Gram matrix representing v_i \cdot v_j
            X = cp.Variable((total_vars, total_vars), PSD=True)

            # Unit vector constraint: Diagonals must be 1 (Euclidean norm = 1)
            constraints = [cp.diag(X) == 1]

            # Formulate the Objective: Maximize sum of (1 + v_u \cdot v_p) / 2
            objective_terms = []
            for u, p in filtered_edges:
                u_idx = neuron_to_idx[u]
                p_idx = num_neurons + proxy_to_idx[p]
                objective_terms.append(0.5 * (1 + X[u_idx, p_idx]))

            objective = cp.Maximize(cp.sum(objective_terms))

            # Partial Core Constraint Without Dummy Neurons
            # sum(v_u * v_w) <= (2 * C_max - N_k)^2
            for c, neurons_in_core_idx in cores_to_neurons_idx.items():
                N_k = len(neurons_in_core_idx)
                if N_k > C_max:
                    # Create an indicator vector to sum only the block of variables for this core
                    indicator = np.zeros(total_vars)
                    indicator[neurons_in_core_idx] = 1.0

                    # cp.quad_form(indicator, X) evaluates exactly to sum_{u,w in core} X_{uw}
                    max_discrepancy_sq = (2 * C_max - N_k) ** 2
                    constraints.append(cp.quad_form(indicator, X) <= max_discrepancy_sq)

            # Solve the continuous relaxation
            problem = cp.Problem(objective, constraints)
            try:
                print(f"  Solving SDP (vars={total_vars}, time limit={self.time_limit_secs}s)...", flush=True)
                problem.solve(solver=cp.SCS, time_limit_secs=self.time_limit_secs)
                X_opt = X.value
                if X_opt is None:
                    raise cp.error.SolverError("SCS solver returned None for variable values.")
                print(f"  SDP solved successfully (status={problem.status}).", flush=True)
            except cp.error.SolverError as e:
                print(f"  SDP solver failed or hit time limit ({e}). Using identity fallback.", flush=True)
                # Mathematical fallback for severe numerical instability
                X_opt = np.eye(total_vars)

                # Recover vectors via Cholesky/Eigendecomposition
            try:
                # Adding tiny epsilon to diagonal to ensure strict positive-definiteness for Cholesky
                X_opt_stable = X_opt + 1e-6 * np.eye(total_vars)
                V = scipy.linalg.cholesky(X_opt_stable, lower=False)
            except scipy.linalg.LinAlgError:
                # Fallback to Eigendecomposition if Cholesky fails due to extreme sparsity
                eigenvalues, eigenvectors = np.linalg.eigh(X_opt)
                eigenvalues[eigenvalues < 0] = 0
                V = eigenvectors @ np.diag(np.sqrt(eigenvalues))

            # =================================================================
            # Phase 3 & 4: The Random Hyperplane Rounding
            # =================================================================
            # Generate multivariate standard normal vector
            r = np.random.randn(total_vars)

            # Project all vectors onto the normal. Vector multiplication yields the scalar projections
            projections = V @ r

            # Raw GW Cut decisions: sign(r * v) mapped to {0, 1}
            y_raw = np.where(projections[:num_neurons] >= 0, 1, 0)
            y = y_raw.astype(float)

            # =================================================================
            # Phase 5: The Bisection Reality Check (Frieze-Jerrum Fix)
            # =================================================================
            for c, neurons_in_core_idx in cores_to_neurons_idx.items():
                nodes_in_core = np.array(neurons_in_core_idx)
                core_y = y[nodes_in_core]

                n_right = int(np.sum(core_y == 1))
                n_left = int(np.sum(core_y == 0))

                # If target bounds are violated, we must project and flip
                excess_right = n_right - R
                excess_left = n_left - L

                if excess_right > 0:
                    # Find the nodes in slice 1 closest to the decision boundary (smallest absolute projection)
                    right_nodes = nodes_in_core[core_y == 1]
                    dists_to_hyperplane = np.abs(projections[right_nodes])
                    # Sort ascending by distance, pick the `excess_right` closest nodes
                    nodes_to_flip = right_nodes[np.argsort(dists_to_hyperplane)[:excess_right]]
                    y[nodes_to_flip] = 0.0

                elif excess_left > 0:
                    # Find the nodes in slice 0 closest to the decision boundary
                    left_nodes = nodes_in_core[core_y == 0]
                    dists_to_hyperplane = np.abs(projections[left_nodes])
                    nodes_to_flip = left_nodes[np.argsort(dists_to_hyperplane)[:excess_left]]
                    y[nodes_to_flip] = 1.0

            # Proxy Resolution: Greedy majority vote based on the fixed discrete neural assignments
            z_rounded = np.zeros(num_proxies, dtype=np.int_)
            proxy_votes = {p_idx: 0 for p_idx in range(num_proxies)}
            for u, p in filtered_edges:
                u_idx = neuron_to_idx[u]
                p_idx = proxy_to_idx[p]
                if y[u_idx] > 0.5:
                    proxy_votes[p_idx] += 1
                else:
                    proxy_votes[p_idx] -= 1

            for p_idx in range(num_proxies):
                z_rounded[p_idx] = 1 if proxy_votes[p_idx] > 0 else 0

        # Zwischenspeichern der binären Slice-Entscheidung
        for p, p_idx in proxy_to_idx.items():
            slice_assignment[p, depth] = z_rounded[p_idx]

        # --- Vorbereitung der nächsten Rekursionsstufe ---
        neurons_left, neurons_right = [], []
        for i, u in enumerate(active_neurons):
            if y[i] < 0.5:
                neurons_left.append(u)
            else:
                neurons_right.append(u)

        edges_left, edges_right = [], []
        if num_edges > 0:
            for u, p in filtered_edges:
                u_idx = neuron_to_idx[u]
                p_idx = proxy_to_idx[p]
                if y[u_idx] < 0.5 and z_rounded[p_idx] == 0:
                    edges_left.append((u, p))
                elif y[u_idx] >= 0.5 and z_rounded[p_idx] == 1:
                    edges_right.append((u, p))

        mid_addr = local_addr_start + L

        # Rekursiver Abstieg
        self.recursive_assign_addresses_and_slices(
            clustering, neurons_left, edges_left, depth + 1,
            local_addr_start, mid_addr, local_addresses, slice_assignment
        )
        self.recursive_assign_addresses_and_slices(
            clustering, neurons_right, edges_right, depth + 1,
            mid_addr, local_addr_end, local_addresses, slice_assignment
        )