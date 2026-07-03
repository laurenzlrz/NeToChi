import math
from typing import Tuple

import graph_tool.all as gt
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
import scipy.sparse as sp

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput


# Annahme: LocalAddressAssigner, ClusterAndHwOutput und DimensionError
# sind wie in deinem Kontext definiert bereits importiert.

class PipageRoundingAssigner(LocalAddressAssigner):
    """
    Weist Neuronen lokale Adressen innerhalb eines Cores zu (0.5-Approximation).
    Das Problem wird als rekursive Bisektion eines bipartiten Sternengraphen formuliert,
    über eine LP-Relaxierung mit anschließendem abhängigem Runden (Pipage Rounding) gelöst.
    """

    def assign_addresses(self, mapping_input: MappingInput, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:
        local_addresses, _ = self.assign_addresses_and_slices(mapping_input, clustering)
        return local_addresses

    def assign_addresses_and_slices(self, mapping_input: MappingInput, clustering: 'ClusterAndHwOutput') -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        graph = mapping_input.graph
        hw = clustering.hw
        num_neurons = clustering.cluster_assignment.shape[0]

        # Initialisierung der Rückgabe-Arrays
        local_addresses = np.full(num_neurons, -1, dtype=np.int_)
        # Spalte 0 bleibt standardmäßig 0 (für d=0).
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

        # Globale Auflösung der Slice-Indizes:
        # Ein Target-Neuron baut seinen absoluten Slice-Index Ebene für Ebene auf.
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
        hw = clustering.hw

        # Abbruchbedingung 1: Keine Neuronen mehr im aktuellen Block
        if len(active_neurons) == 0:
            return

        # Abbruchbedingung 2: Router-Level-Tiefe erreicht oder Blockgröße = 1
        # Die Kapazitätsgrenzen wurden durch die Rekursion streng eingehalten,
        # wir können die verbleibenden Adressen sequentiell zuweisen.
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

        # Kanten filtern: Nur Proxy-Entscheidungen betrachten, die auf dieser Tiefe oder tiefer relevant sind
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
        L = math.ceil((local_addr_end - local_addr_start) / 2.0)
        R = (local_addr_end - local_addr_start) - L

        # Mapping: Welche aktiven Neuronen liegen in welchem Core
        cores_to_neurons_idx = {}
        for i, u in enumerate(active_neurons):
            c = clustering.cluster_assignment[u]
            cores_to_neurons_idx.setdefault(c, []).append(i)

        # Fallback: Wenn es keine relevanten Kanten mehr gibt, einfach in der Mitte teilen
        if num_edges == 0:
            y = np.zeros(num_neurons)
            for c, neurons_in_core_idx in cores_to_neurons_idx.items():
                for i, u_idx in enumerate(neurons_in_core_idx):
                    if i >= L:
                        y[u_idx] = 1.0
            z_rounded = np.zeros(0, dtype=np.int_)
        else:
            # --- LP Aufstellen ---
            total_vars = num_neurons + num_proxies + num_edges
            c_obj = np.zeros(total_vars)
            c_obj[num_neurons + num_proxies:] = -1.0

            bounds = [(0.0, 1.0) for _ in range(total_vars)]
            num_active_cores = len(cores_to_neurons_idx)

            A_ub = sp.lil_matrix((2 * num_edges + 2 * num_active_cores, total_vars))
            b_ub = np.zeros(2 * num_edges + 2 * num_active_cores)

            row_idx = 0
            for e_idx, (u, p) in enumerate(filtered_edges):
                x_col = num_neurons + num_proxies + e_idx
                u_idx = neuron_to_idx[u]
                p_idx = num_neurons + proxy_to_idx[p]

                A_ub[row_idx, x_col] = 1
                A_ub[row_idx, u_idx] = 1
                A_ub[row_idx, p_idx] = -1
                b_ub[row_idx] = 1
                row_idx += 1

                A_ub[row_idx, x_col] = 1
                A_ub[row_idx, u_idx] = -1
                A_ub[row_idx, p_idx] = 1
                b_ub[row_idx] = 1
                row_idx += 1

            for c, neurons_in_core_idx in cores_to_neurons_idx.items():
                for u_idx in neurons_in_core_idx:
                    A_ub[row_idx, u_idx] = 1
                b_ub[row_idx] = R
                row_idx += 1

                for u_idx in neurons_in_core_idx:
                    A_ub[row_idx, u_idx] = -1
                b_ub[row_idx] = L - len(neurons_in_core_idx)
                row_idx += 1

            res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                assert False
                # Mathematisches Fallback, falls der Solver bei Extremlasten numerisch instabil wird
                y = np.zeros(num_neurons)
                for c, neurons_in_core_idx in cores_to_neurons_idx.items():
                    for i, u_idx in enumerate(neurons_in_core_idx):
                        if i >= L: y[u_idx] = 1.0
                z = np.zeros(num_proxies)
            else:
                y = res.x[:num_neurons].copy()
                z = res.x[num_neurons:num_neurons + num_proxies].copy()

            # --- Pipage Rounding ---
            d = np.zeros(num_neurons)
            neuron_neighbors = {i: [] for i in range(num_neurons)}
            for u, p in filtered_edges:
                neuron_neighbors[neuron_to_idx[u]].append(proxy_to_idx[p])
            for i in range(num_neurons):
                for p_idx in neuron_neighbors[i]:
                    d[i] += (2 * z[p_idx] - 1)

            for c, neurons_in_core_idx in cores_to_neurons_idx.items():
                eps_tol = 1e-6
                fractional_nodes = [i for i in neurons_in_core_idx if eps_tol < y[i] < 1.0 - eps_tol]

                while len(fractional_nodes) > 0:
                    if len(fractional_nodes) >= 2:
                        u_idx = fractional_nodes.pop()
                        v_idx = fractional_nodes.pop()
                        self._round_variable_pair(u_idx, v_idx, y, d[u_idx], d[v_idx])
                        if eps_tol < y[u_idx] < 1.0 - eps_tol: fractional_nodes.append(u_idx)
                        if eps_tol < y[v_idx] < 1.0 - eps_tol: fractional_nodes.append(v_idx)
                    else:
                        w_idx = fractional_nodes.pop()
                        self._round_single_variable(w_idx, y, d[w_idx])

            # Proxy-Resolution (Greedy Majority Vote)
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

        # Zwischenspeichern der binären Slice-Entscheidung auf dieser Ebene
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
                # Wenn Kante komplett in der linken/rechten Hälfte bleibt, überlebt sie
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

    def _round_variable_pair(self, u: int, v: int, y: np.ndarray, d_u: float, d_v: float) -> None:
        """
        Führt die Verschiebung (Perturbation) für zwei fraktionale Variablen im selben Core durch.
        """
        A = d_u - d_v
        eps_max = min(1.0 - y[u], y[v])
        eps_min = max(-y[u], y[v] - 1.0)

        w_delta_max = A * eps_max
        w_delta_min = A * eps_min

        if w_delta_max >= w_delta_min:
            best_eps = eps_max
        else:
            best_eps = eps_min

        y[u] += best_eps
        y[v] -= best_eps

        y[u] = np.clip(y[u], 0.0, 1.0)
        y[v] = np.clip(y[v], 0.0, 1.0)

        if np.isclose(y[u], 0.0): y[u] = 0.0
        if np.isclose(y[u], 1.0): y[u] = 1.0
        if np.isclose(y[v], 0.0): y[v] = 0.0
        if np.isclose(y[v], 1.0): y[v] = 1.0

    def _round_single_variable(self, w: int, y: np.ndarray, d_w: float) -> None:
        """
        Behandelt den Randfall: Eine einzige fraktionale Variable verbleibt im Core.
        """
        if d_w > 0:
            y[w] = 1.0
        else:
            y[w] = 0.0