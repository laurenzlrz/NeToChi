"""
Co-fan-in address assigner (LocalAddressAssigner): assigns each neuron a local index
within its core so that neurons sharing targets get shared index prefixes.

Idea
----
Slices are dyadic index ranges: a source u is valid for target q at distance d exactly
if the first d bits of its index equal the slice index k_d(q). Two sources that share
an external target should therefore have long common index prefixes.

Per core:
  1. S[u, u'] = number of external targets (targets in other cores) shared by u and u'.
     Edges inside a core are always valid and are ignored.
  2. Recursive bisection of the index range [lo, hi): split it in the middle, sort the
     nodes by the Fiedler vector of S restricted to them, and give the first half of
     them to the left half of the range, the rest to the right half.
  3. Recurse until every node has its own position. Free slots end up spread out.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

# ----------------------------------------------------------------------------
# ADAPT these imports to your project layout
# ----------------------------------------------------------------------------
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput  # ADAPT
from netochi.mapping.three_step_mapping.clustering.clusterer.spectral_tree_clusterer import fiedler_vector


class CoFanInAddressAssigner(LocalAddressAssigner):
    """Recursive spectral bisection of every core's index range on co-fan-in."""

    def assign_addresses(self, mapping_input: MosaicMappingInput,
                         clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:
        """neuron_id -> local index within its core (unique within each core)."""
        hw = clustering.hw
        core = np.asarray(clustering.cluster_assignment, dtype=np.int64)
        n = len(core)
        per_core = hw.neurons_per_core

        graph = mapping_input.graph
        edges = np.asarray(graph.get_edges())[:, :2].astype(np.int64).reshape(-1, 2)
        src, dst = edges[:, 0], edges[:, 1]

        rng = np.random.default_rng(mapping_input.seed)

        ext = core[src] != core[dst]          # only edges between different cores matter
        A = sp.csr_matrix((np.ones(ext.sum()), (src[ext], dst[ext])), shape=(n, n))

        idx = np.zeros(n, dtype=np.int64)
        for c in range(hw.total_cores):
            nodes = np.flatnonzero(core == c)
            if nodes.size == 0:
                continue
            if nodes.size > per_core:
                raise ValueError(f"core {c} holds {nodes.size} nodes, capacity is {per_core}")
            S = (A[nodes] @ A[nodes].T).toarray()  # shared external targets

            def rec(loc: np.ndarray, lo: int, hi: int) -> None:
                if len(loc) == 0:
                    return
                if len(loc) == 1 or hi - lo == 1:
                    idx[nodes[loc]] = lo + np.arange(len(loc))
                    return
                mid, m = (lo + hi) // 2, len(loc)
                m_l = int(np.clip(-(-m // 2), max(0, m - (hi - mid)), min(m, mid - lo)))
                order = np.argsort(fiedler_vector(S[np.ix_(loc, loc)], rng))
                rec(loc[order[:m_l]], lo, mid)
                rec(loc[order[m_l:]], mid, hi)

            rec(np.arange(len(nodes)), 0, per_core)
        return idx