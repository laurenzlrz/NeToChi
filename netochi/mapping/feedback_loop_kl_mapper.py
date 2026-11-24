"""
Feedback-loop mapper with a pairwise Kernighan-Lin core step.

Identical to ``FeedbackLoopMapper`` (slice step, index step, edge-weight feedback, restarts),
except for step C. Instead of pain-guided greedy moves/swaps, the core step is a pairwise
Kernighan-Lin refinement of the same weighted distance-cost proxy:

  for every core a and its `kl_partners` most strongly connected partner cores b:
      repeat (at most `kl_max_passes` times):
          swap the best pair (u in a, v in b) among the unlocked neurons, even if the proxy
          gets worse, lock u and v, and continue until one side is exhausted;
          keep only the best prefix of that swap sequence (nothing if no prefix improves)

Because it may pass through worse states, it can move a whole group of neurons from one core to
the other. This repairs planted cores that ended up split over two cores, where every single swap
is neutral or worse and the greedy core step gets stuck. As before, a core step that raises the
true loss is undone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from netochi.mapping.feedback_loop_mapper import FeedbackLoopConfig, FeedbackLoopMapper, FeedbackLoopSolver


@dataclass
class FeedbackLoopKLConfig(FeedbackLoopConfig):
    # Note: core_attempts_factor, swap_candidates and target_cores only affect the greedy core
    # step of FeedbackLoopMapper and are ignored here.
    kl_partners: int = 2               # per core: refine with its most strongly connected partner cores
    kl_max_passes: int = 2             # KL passes per core pair (stops early without improvement)

    def create(self) -> "FeedbackLoopKLMapper":
        return FeedbackLoopKLMapper(config=self)


class FeedbackLoopKLSolver(FeedbackLoopSolver):
    """FeedbackLoopSolver whose core step is the pairwise KL refinement."""

    cfg: FeedbackLoopKLConfig

    def _core_step(self, core, idx, w, viol) -> int:
        return self._kl_refine(core, idx, w)

    def _kl_refine(self, core, idx, w) -> int:
        """
        Kernighan-Lin refinement of the proxy objective sum_e w_e * cost(dist(e)), applied to
        pairs of cores. Returns the number of swaps kept.
        """
        cfg, n, W = self.cfg, self.n, self.W
        Wm = sp.csr_matrix((w, (self.src, self.dst)), shape=(n, n))
        Wm = (Wm + Wm.T).tocsr()
        Wm.setdiag(0.0)
        Wm.eliminate_zeros()
        Cd = self.cost[self.D]

        onehot = sp.csr_matrix((np.ones(n), (np.arange(n), core)), shape=(n, W))
        between = (onehot.T @ Wm @ onehot).toarray()
        np.fill_diagonal(between, -np.inf)
        pairs = set()
        for a in range(W):
            for b in np.argsort(-between[a])[:cfg.kl_partners]:
                if np.isfinite(between[a, b]) and between[a, b] > 0:
                    pairs.add((min(a, int(b)), max(a, int(b))))

        kept = 0
        for a, b in sorted(pairs, key=lambda p: -between[p]):
            for _ in range(cfg.kl_max_passes):
                swaps = self._kl_pass(core, Wm, Cd, a, b)
                if not swaps:
                    break
                for u, v in swaps:
                    core[u], core[v] = b, a
                    idx[u], idx[v] = idx[v], idx[u]
                kept += len(swaps)
        return kept

    def _kl_pass(self, core, Wm, Cd, a, b) -> List[Tuple[int, int]]:
        """One KL pass between cores a and b; returns the swaps of the best prefix."""
        A = np.flatnonzero(core == a)
        B = np.flatnonzero(core == b)
        if A.size == 0 or B.size == 0:
            return []
        nodes = np.concatenate([A, B])
        sub = Wm[nodes][:, nodes].toarray()                  # weights among the pair's neurons
        # cost[t, c] = sum_j W_tj Cd[c, core_j]; only columns a and b are needed here
        nbw = (Wm[nodes] @ sp.csr_matrix((np.ones(self.n), (np.arange(self.n), core)),
                                         shape=(self.n, self.W))).toarray()
        cost_a, cost_b = nbw @ Cd[a], nbw @ Cd[b]
        cab = Cd[a, b] - Cd[a, a]                            # proxy cost of a pair split over a, b

        on_a = np.zeros(nodes.size, dtype=bool)
        on_a[:A.size] = True
        free = np.ones(nodes.size, dtype=bool)
        seq: List[Tuple[int, int]] = []
        total, best_total, best_len = 0.0, -1e-9, 0
        for _ in range(min(A.size, B.size)):
            ia = np.flatnonzero(free & on_a)
            ib = np.flatnonzero(free & ~on_a)
            if ia.size == 0 or ib.size == 0:
                break
            g_a = cost_b[ia] - cost_a[ia]                    # change if u moves a -> b
            g_b = cost_a[ib] - cost_b[ib]                    # change if v moves b -> a
            delta = g_a[:, None] + g_b[None, :] + 2.0 * cab * sub[np.ix_(ia, ib)]
            r, c = np.unravel_index(int(np.argmin(delta)), delta.shape)
            iu, iv = ia[r], ib[c]
            total += float(delta[r, c])
            seq.append((int(nodes[iu]), int(nodes[iv])))
            free[iu] = free[iv] = False
            # u now sits on b, v on a: update the costs of all pair neurons
            diff = (sub[:, iu] - sub[:, iv]) * cab
            cost_a += diff
            cost_b -= diff
            if total < best_total:
                best_total, best_len = total, len(seq)
        return seq[:best_len]


class FeedbackLoopKLMapper(FeedbackLoopMapper):
    """FeedbackLoopMapper with the pairwise Kernighan-Lin core step (see module docstring)."""

    solver_class = FeedbackLoopKLSolver

    def __init__(self, config: FeedbackLoopKLConfig | None = None) -> None:
        super().__init__(config or FeedbackLoopKLConfig())

    def get_name(self) -> str:
        name = f"FeedbackKL_{self.clusterer.get_name()}_{self.address_assigner.get_name()}"
        if self.config.restarts > 1:
            name += f"_R{self.config.restarts}"
        return name
