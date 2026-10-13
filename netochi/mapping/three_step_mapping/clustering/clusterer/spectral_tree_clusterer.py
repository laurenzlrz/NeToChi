"""
Spectral tree clusterer: assigns graph nodes to cores by recursive bisection that
follows the routing tree (ClustererFixedHw).

Idea
----
The cores are the leaves of the routing tree. At each recursion step, the current
set of cores is split into two groups of whole subtrees, and the nodes are split
accordingly: the number of nodes per group is proportional to its capacity, and the
Fiedler vector of the graph restricted to the current nodes decides WHICH nodes go
left or right (strongly connected nodes get similar values, so cutting the sorted
order cuts few edges). Heavy cuts therefore happen low in the tree, where they are
cheapest. Core capacities are always respected.

Output convention (ClusterAndHwOutput): cluster id == core id, i.e. cluster ids are
numbered like the cores on the hardware, and num_clusters == hw.total_cores
(some clusters may be empty if the graph has fewer nodes than hardware neurons).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# ----------------------------------------------------------------------------
# ADAPT these imports to your project layout
# ----------------------------------------------------------------------------
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw, ClusterAndHwOutput   # ADAPT


def fiedler_vector(S, rng: np.random.Generator) -> np.ndarray:
    """Fiedler vector of a symmetric non-negative similarity matrix (dense or sparse)."""
    m = S.shape[0]
    if m <= 2:
        return rng.random(m)
    S = sp.csr_matrix(S)
    S = S - sp.diags(S.diagonal())
    deg = np.asarray(S.sum(axis=1)).ravel()
    if m <= 1500:
        lap = np.diag(deg) - S.toarray()
        vec = np.linalg.eigh(lap)[1][:, 1]
    else:  # power iteration on (c*I - L), constant vector deflated
        c = 2.0 * deg.max() + 1e-9
        vec = rng.standard_normal(m)
        for _ in range(300):
            vec -= vec.mean()
            vec = c * vec - (deg * vec - S @ vec)
            vec /= np.linalg.norm(vec) + 1e-12
    return vec + 1e-9 * rng.standard_normal(m)


class SpectralTreeClusterer(ClustererFixedHw[MosaicMappingInput]):
    """Recursive spectral bisection of the graph along the routing tree."""

    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        hw = input_data.hw_config
        graph = input_data.graph
        n = int(graph.num_vertices())
        n_cores, per_core = hw.total_cores, hw.neurons_per_core
        if n > n_cores * per_core:
            raise ValueError(f"{n} nodes do not fit on {n_cores * per_core} hardware neurons")

        edges = np.asarray(graph.get_edges())[:, :2].astype(np.int64).reshape(-1, 2)
        A = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        asym = A + A.T

        dist = np.array([[hw.core_distance(a, b) for b in range(n_cores)]
                         for a in range(n_cores)], dtype=np.int64)

        rng = np.random.default_rng(input_data.seed)
        core = np.zeros(n, dtype=np.int64)
        self._partition(np.arange(n), np.arange(n_cores), core, dist, asym, per_core, rng)

        return ClusterAndHwOutput(cluster_assignment=core, num_clusters=n_cores, hw=hw)

    def _partition(self, nodes, cores, core, dist, asym, per_core, rng) -> None:
        if len(nodes) == 0:
            return
        if len(cores) == 1:
            core[nodes] = cores[0]
            return

        # child subtrees = connected components of "distance <= lvl - 1"
        sub = dist[np.ix_(cores, cores)]
        lvl = sub.max()
        ncomp, lab = connected_components(sp.csr_matrix(sub <= lvl - 1), directed=False)
        groups = [cores[lab == g] for g in range(ncomp)]
        if ncomp == 1:  # should not happen for a tree; split arbitrarily
            groups = [cores[: len(cores) // 2], cores[len(cores) // 2:]]

        half = len(groups) // 2
        left, right = np.concatenate(groups[:half]), np.concatenate(groups[half:])
        cap_l, cap_r = len(left) * per_core, len(right) * per_core
        m = len(nodes)
        m_l = int(np.clip(round(m * cap_l / (cap_l + cap_r)), max(0, m - cap_r), min(m, cap_l)))

        order = np.argsort(fiedler_vector(asym[nodes][:, nodes], rng))
        self._partition(nodes[order[:m_l]], left, core, dist, asym, per_core, rng)
        self._partition(nodes[order[m_l:]], right, core, dist, asym, per_core, rng)