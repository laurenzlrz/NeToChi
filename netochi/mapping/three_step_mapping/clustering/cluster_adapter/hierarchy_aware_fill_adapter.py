"""
Hierarchy-aware fill adapter (ClusteringAdapterFixedHw).

Problem
-------
A clustering algorithm returns a hierarchy of clusters whose sizes do not have to match
the cores of the (fixed) hardware. The adapter must place every neuron on a core, never
exceeding `neurons_per_core`, while keeping related neurons close in the routing tree.

Idea
----
1. Put the neurons into one sequence in depth-first order of the cluster hierarchy:
   neurons of the same leaf cluster are consecutive, sibling clusters are next to each
   other, and so on. Closely related neurons are therefore close in the sequence.
2. Give every boundary between two consecutive neurons a "closeness" = depth of the
   lowest common ancestor of their leaf clusters (a boundary inside one leaf cluster
   gets the highest value). Cutting at a boundary with high closeness separates
   closely related neurons.
3. Cut the sequence top-down along the routing tree. At each router, the sequence is
   split into contiguous pieces, one per child subtree, each fitting the subtree's
   capacity. Among all cut positions that keep the remaining pieces feasible, the cut
   with the LOWEST closeness is chosen (ties: closest to a capacity-proportional
   position, so slack is spread over the subtrees). The top-level cuts, which create the
   largest routing distances, therefore get the first choice of the weakest boundaries.
4. Recurse until every piece is a single core.

Properties
----------
* Core capacities are always respected (raises HardwareConstraintError on overflow).
* A leaf cluster that fits into one core is only split if the slack does not allow
  keeping it intact; a cluster larger than a core is split into adjacent cores.
* Sibling clusters end up in the same or nearby subtrees.
* Works without a hierarchy (cluster_parent missing): then only leaf clusters are kept
  intact where possible.

Output convention: cluster id == core id, num_clusters == hw.total_cores.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.definitions.exceptions import HardwareConstraintError
from netochi.mapping.three_step_mapping.interfaces import (
    HierarchicalClusterOutput, ClusterAndHwOutput, ClusteringAdapterFixedHw,
)


class HierarchyAwareFillAdapter(ClusteringAdapterFixedHw):
    """Fills the fixed hardware top-down along the routing tree, cutting the
    depth-first neuron order where the cluster hierarchy is weakest."""

    def get_name(self) -> str:
        return "HierarchyFill"

    # ------------------------------------------------------------------ public
    def adapt_clustering(self, clustering: HierarchicalClusterOutput,
                         hw_config: MosaicHardwareConfig) -> ClusterAndHwOutput:
        leaf_of = np.asarray(clustering.cluster_assignment, dtype=np.int64)
        n = len(leaf_of)
        n_cores, per_core = hw_config.total_cores, hw_config.neurons_per_core
        if n > n_cores * per_core:
            raise HardwareConstraintError(
                f"Hardware overflow: {n} neurons, but only {n_cores * per_core} slots on chip.")

        order, closeness = self._dfs_order(clustering, leaf_of)

        dist = np.array([[hw_config.core_distance(a, b) for b in range(n_cores)]
                         for a in range(n_cores)], dtype=np.int64)
        core_of = np.zeros(n, dtype=np.int64)
        self._place(0, n, np.arange(n_cores), order, closeness, core_of, dist, per_core)

        return ClusterAndHwOutput(cluster_assignment=core_of.astype(np.int_),
                                  num_clusters=n_cores, hw=hw_config)

    # ------------------------------------------------------------------ step 1+2
    def _dfs_order(self, clustering: HierarchicalClusterOutput, leaf_of: np.ndarray):
        """Neuron order (DFS over the hierarchy) and closeness of each boundary.

        closeness[i] belongs to the boundary between order[i-1] and order[i]
        (closeness[0] is unused)."""
        neurons_of: Dict[int, List[int]] = defaultdict(list)
        for v, leaf in enumerate(leaf_of):
            neurons_of[int(leaf)].append(v)

        parent = getattr(clustering, "cluster_parent", None)
        children: Dict[int, List[int]] = defaultdict(list)
        roots: List[int] = []
        if parent is not None and len(parent) > 0:
            parent = np.asarray(parent, dtype=np.int64)
            for cid, p in enumerate(parent):
                if p < 0:
                    roots.append(cid)
                else:
                    children[int(p)].append(cid)
        # leaves that are not part of the hierarchy (or no hierarchy at all)
        in_tree = set(range(len(parent))) if parent is not None else set()
        loose = sorted(l for l in neurons_of if l not in in_tree)

        # DFS; path[c] = list of clusters from the root to c (for LCA depths)
        order: List[int] = []
        leaf_path: Dict[int, List[int]] = {}
        stack = [(r, [r]) for r in sorted(roots, reverse=True)]
        while stack:
            cid, path = stack.pop()
            if cid in neurons_of:                       # a leaf with neurons
                leaf_path[cid] = path
                order.extend(neurons_of[cid])
            for ch in sorted(children.get(cid, []), reverse=True):
                stack.append((ch, path + [ch]))
        for l in loose:                                 # flat fallback
            leaf_path[l] = [-1, l]
            order.extend(neurons_of[l])

        order_arr = np.array(order, dtype=np.int64)
        assert len(order_arr) == len(leaf_of) and len(set(order)) == len(order), \
            "every neuron must belong to exactly one leaf cluster"

        max_depth = max((len(p) for p in leaf_path.values()), default=1)
        closeness = np.zeros(len(order_arr), dtype=np.int64)
        for i in range(1, len(order_arr)):
            la, lb = int(leaf_of[order_arr[i - 1]]), int(leaf_of[order_arr[i]])
            if la == lb:
                closeness[i] = max_depth + 1            # inside one cluster: never cut if avoidable
            else:
                pa, pb = leaf_path[la], leaf_path[lb]
                k = 0
                while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
                    k += 1
                closeness[i] = k                        # depth of the lowest common ancestor
        return order_arr, closeness

    # ------------------------------------------------------------------ step 3+4
    def _subtree_groups(self, cores: np.ndarray, dist: np.ndarray) -> List[np.ndarray]:
        """Child subtrees of the router above `cores` (ordered by core id)."""
        sub = dist[np.ix_(cores, cores)]
        lvl = sub.max()
        ncomp, lab = connected_components(sp.csr_matrix(sub <= lvl - 1), directed=False)
        groups = [np.sort(cores[lab == g]) for g in range(ncomp)]
        groups.sort(key=lambda g: g[0])
        if len(groups) == 1:                            # degenerate: split in halves
            h = len(cores) // 2
            groups = [cores[:h], cores[h:]]
        return groups

    def _place(self, lo: int, hi: int, cores: np.ndarray, order: np.ndarray,
               closeness: np.ndarray, core_of: np.ndarray, dist: np.ndarray,
               per_core: int) -> None:
        """Assign the neurons order[lo:hi] to the subtree `cores`."""
        m = hi - lo
        if m == 0:
            return
        if len(cores) == 1:
            core_of[order[lo:hi]] = cores[0]
            return

        groups = self._subtree_groups(cores, dist)
        caps = [len(g) * per_core for g in groups]
        total_cap = sum(caps)

        start = lo
        for gi, g in enumerate(groups):
            if gi == len(groups) - 1:
                end = hi
            else:
                rest_cap = sum(caps[gi + 1:])
                remaining = hi - start
                e_min = start + max(0, remaining - rest_cap)   # the others must still fit
                e_max = start + min(remaining, caps[gi])        # this subtree must fit
                target = start + round(remaining * caps[gi] / (caps[gi] + rest_cap))
                cand = np.arange(e_min, e_max + 1)
                # boundary cost; cutting at the ends of [lo, hi) costs nothing new
                cost = np.array([0 if (e == lo or e == hi) else closeness[e] for e in cand])
                best = cand[np.lexsort((np.abs(cand - target), cost))[0]]
                end = int(best)
            self._place(start, end, g, order, closeness, core_of, dist, per_core)
            start = end
