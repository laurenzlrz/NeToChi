"""
Feedback-loop mapper for the Mosaic hardware (edge-weight feedback).

Pipeline
--------
init : cores   <- ClustererFixedHw (default: SpectralTreeClusterer, recursive spectral
                         bisection along the routing tree)
       indices <- LocalAddressAssigner (default: CoFanInAddressAssigner, recursive bisection
                         of each core's index range on co-fan-in)
repeat:
  A. slice step   (exact)     per (target q, distance d): pick the fullest slice
  B. index step   (exact)     per core: linear assignment with slices fixed
  A. slice step               re-fit slices to the new indices, then measure loss
  D. feedback                 violated edges get w *= (1+eta), satisfied w *= gamma
  C. core step    (heuristic) pain-guided moves/swaps between cores, evaluated
                              with the edge weights on a distance-cost proxy

The true loss (number of violated edges) is only measured, never optimised
directly. The best mapping seen so far is returned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------------
# ADAPT these imports to your project layout
# ----------------------------------------------------------------------------
from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment
from netochi.mapping.interfaces import BaseMapper                      # ADAPT
from netochi.mapping.interfaces import MosaicNetworkMappingState           # ADAPT
from netochi.mapping.three_step_mapping.clustering.clusterer.spectral_tree_clusterer import \
    SpectralTreeClusterer
from netochi.mapping.three_step_mapping.local_address_assignment.cofanin_address_assigner import \
    CoFanInAddressAssigner                                              # ADAPT
from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw, LocalAddressAssigner


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class FeedbackLoopConfig:
    max_iters: int = 60
    patience: int = 15                 # stop after this many iterations without a new best
    time_limit_s: Optional[float] = None
    inner_rounds: int = 2              # (slice, index) coordinate-descent rounds per iteration
    core_every: int = 2                # expensive core step every k-th iteration
                                       # (a core step is undone if it raises the true loss)
    core_attempts_factor: float = 0.2  # attempted moves per core step = factor * n
    swap_candidates: int = 12          # partners evaluated per target core
    target_cores: int = 3              # best target cores tried per node
    eta: float = 0.3                   # upweight factor for violated edges
    gamma: float = 0.9                 # decay factor for satisfied edges
    w_min: float = 0.2                 # weight caps: hopeless edges must not dominate
    w_max: float = 10.0
    weighted_index_step: bool = True   # use edge weights in the index step
    distance_cost: Optional[Callable[[int], float]] = None  # proxy cost per distance
    verbose: bool = False
    # clusterer producing the initial core assignment (cluster id == core id)
    core_initializer: ClustererFixedHw = SpectralTreeClusterer()
    # assigner producing the initial local indices (given the clustering)
    address_initializer: LocalAddressAssigner = CoFanInAddressAssigner()

    def create(self) -> "FeedbackLoopMapper":
        return FeedbackLoopMapper(config=self)


def _default_distance_cost(d: int) -> float:
    # 0 inside a core; 1 - 2^-d = probability that a random source index is NOT
    # inside the target's slice at distance d.
    return 0.0 if d == 0 else 1.0 - 2.0 ** (-d)


# ----------------------------------------------------------------------------
# Solver (pure numpy, independent of graph_tool)
# ----------------------------------------------------------------------------
class FeedbackLoopSolver:
    def __init__(self, hw, src: np.ndarray, dst: np.ndarray, n: int,
                 cfg: FeedbackLoopConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.src, self.dst, self.n, self.E = src, dst, n, len(src)

        self.W = hw.total_cores
        self.N = hw.neurons_per_core
        self.L = hw.router_levels
        if n > self.W * self.N:
            raise ValueError(f"{n} nodes do not fit on {self.W * self.N} hardware neurons")

        self.D = np.array([[hw.core_distance(a, b) for b in range(self.W)]
                           for a in range(self.W)], dtype=np.int64)
        self.ns = np.array([hw.num_slices_at_distance(d) for d in range(self.L + 1)], dtype=np.int64)
        self.size = np.array([hw.get_slice_bounds(d, 0)[1] - hw.get_slice_bounds(d, 0)[0]
                              for d in range(self.L + 1)], dtype=np.int64)
        assert np.all(self.ns * self.size == self.N), "slices must tile the core"

        fn = cfg.distance_cost or _default_distance_cost
        self.cost = np.array([fn(d) for d in range(self.L + 1)], dtype=float)


    # ------------------------------------------------------------------ helpers
    def _violated(self, core, idx, k) -> np.ndarray:
        d = self.D[core[self.src], core[self.dst]]
        s = idx[self.src] // self.size[d]
        return s != k[self.dst, d]

    # ------------------------------------------------------------------ A: slices
    def _slice_step(self, core, idx, k) -> None:
        d_e = self.D[core[self.src], core[self.dst]]
        s_e = idx[self.src] // self.size[d_e]
        rows = np.arange(self.n)
        for d in range(1, self.L + 1):
            m = d_e == d
            ns = self.ns[d]
            hist = np.bincount(self.dst[m] * ns + s_e[m], minlength=self.n * ns)
            hist = hist.reshape(self.n, ns).astype(float)
            hist[rows, k[:, d]] += 0.5          # ties keep the previous slice
            k[:, d] = hist.argmax(axis=1)
        k[:, 0] = 0

    # ------------------------------------------------------------------ B: indices
    def _index_step(self, core, idx, k, w) -> None:
        d_e = self.D[core[self.src], core[self.dst]]
        match = np.zeros((self.n, self.N))
        for d in range(1, self.L + 1):
            m = d_e == d
            ns = self.ns[d]
            key = self.src[m] * ns + k[self.dst[m], d]
            cnt = np.bincount(key, weights=w[m], minlength=self.n * ns).reshape(self.n, ns)
            match += np.repeat(cnt, self.size[d], axis=1)
        match[np.arange(self.n), idx] += 1e-3   # stability
        for c in range(self.W):
            nodes = np.flatnonzero(core == c)
            if nodes.size:
                r, col = linear_sum_assignment(match[nodes], maximize=True)
                idx[nodes[r]] = col

    # ------------------------------------------------------------------ D: feedback
    def _update_weights(self, w, viol) -> np.ndarray:
        c = self.cfg
        w = np.where(viol, w * (1.0 + c.eta), w * c.gamma)
        return np.clip(w, c.w_min, c.w_max)

    # ------------------------------------------------------------------ C: cores
    def _core_step(self, core, idx, w, viol) -> int:
        cfg, rng, n, W, N = self.cfg, self.rng, self.n, self.W, self.N

        Wm = sp.csr_matrix((w, (self.src, self.dst)), shape=(n, n))
        Wm = Wm + Wm.T
        Wm = Wm - sp.diags(Wm.diagonal())
        Wm.eliminate_zeros()
        indptr, indices, data = Wm.indptr, Wm.indices, Wm.data

        wv = w * viol
        pain = np.bincount(self.src, weights=wv, minlength=n) + \
               np.bincount(self.dst, weights=wv, minlength=n)
        if pain.sum() <= 0:
            return 0

        Cd = self.cost[self.D]                      # (W, W) proxy cost per core pair
        owner = -np.ones((W, N), dtype=np.int64)
        owner[core, idx] = np.arange(n)

        attempts = max(1, int(cfg.core_attempts_factor * n))
        picks = rng.choice(n, size=attempts, p=pain / pain.sum())
        moves = 0
        for u in picks:
            a = core[u]
            s, e = indptr[u], indptr[u + 1]
            if s == e:
                continue
            nbw = np.bincount(core[indices[s:e]], weights=data[s:e], minlength=W)
            cost_vec = Cd @ nbw                      # cost of u at every core
            cur = cost_vec[a]

            best_delta, best = -1e-9, None
            tried = 0
            for b in np.argsort(cost_vec):
                if b == a:
                    continue
                gain_u = cost_vec[b] - cur
                if tried >= cfg.target_cores or gain_u >= 0:
                    break
                tried += 1
                free = np.flatnonzero(owner[b] < 0)
                if free.size:                        # plain move into a free slot
                    if gain_u < best_delta:
                        best_delta, best = gain_u, (b, -1, int(free[0]))
                    continue
                occ = owner[b]
                cand = occ if len(occ) <= cfg.swap_candidates else \
                    rng.choice(occ, cfg.swap_candidates, replace=False)
                diff = Cd[a] - Cd[b]
                for v in cand:
                    s2, e2 = indptr[v], indptr[v + 1]
                    gain_v = 0.0
                    if s2 != e2:
                        nbwv = np.bincount(core[indices[s2:e2]], weights=data[s2:e2], minlength=W)
                        gain_v = float(diff @ nbwv)
                    # the u-v edge keeps its distance, remove its double counting
                    delta = gain_u + gain_v + 2.0 * Wm[u, v] * Cd[a, b]
                    if delta < best_delta:
                        best_delta, best = delta, (b, int(v), int(idx[v]))

            if best is None:
                continue
            b, v, j = best
            iu = idx[u]
            if v < 0:                                # move
                owner[a, iu] = -1
                owner[b, j] = u
                core[u], idx[u] = b, j
            else:                                    # swap
                owner[a, iu] = v
                owner[b, j] = u
                core[v], idx[v] = a, iu
                core[u], idx[u] = b, j
            moves += 1
        return moves

    # ------------------------------------------------------------------ main loop
    def solve(self, init_core: np.ndarray, init_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        cfg = self.cfg
        t0 = time.time()
        core = np.asarray(init_core, dtype=np.int64).copy()
        idx = np.asarray(init_idx, dtype=np.int64).copy()
        k = np.zeros((self.n, self.L + 1), dtype=np.int64)
        w = np.ones(self.E)

        history: List[int] = []
        best = None
        stall = 0
        snap = None  # (loss, core, idx, k) taken right before a core step
        for it in range(cfg.max_iters):
            for _ in range(cfg.inner_rounds):
                self._slice_step(core, idx, k)                       # A
                self._index_step(core, idx, k, w if cfg.weighted_index_step else np.ones(self.E))  # B
            self._slice_step(core, idx, k)                           # A (measure)
            viol = self._violated(core, idx, k)
            loss = int(viol.sum())
            if snap is not None and loss > snap[0]:
                # the core step made the true loss worse: undo it, keep the new weights
                loss, core, idx, k = snap[0], snap[1].copy(), snap[2].copy(), snap[3].copy()
                viol = self._violated(core, idx, k)
            snap = None
            history.append(loss)
            if cfg.verbose:
                print(f"iter {it:3d}  loss {loss}")

            if best is None or loss < best[0]:
                best = (loss, core.copy(), idx.copy(), k.copy())
                stall = 0
            else:
                stall += 1
            if loss == 0 or stall >= cfg.patience or \
                    (cfg.time_limit_s and time.time() - t0 > cfg.time_limit_s):
                break

            w = self._update_weights(w, viol)                        # D
            if (it + 1) % cfg.core_every == 0:
                snap = (loss, core.copy(), idx.copy(), k.copy())
                self._core_step(core, idx, w, viol)                  # C
        _, bc, bi, bk = best
        return bc, bi, bk, history


# ----------------------------------------------------------------------------
# Mapper wrapper
# ----------------------------------------------------------------------------
class FeedbackLoopMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """Iterative core / index / slice mapper with multiplicative edge-weight feedback."""

    def __init__(self, config: Optional[FeedbackLoopConfig] = None) -> None:
        self.config = config or FeedbackLoopConfig()
        self.clusterer: ClustererFixedHw = self.config.core_initializer
        self.address_assigner: LocalAddressAssigner = self.config.address_initializer
        self.last_history: List[int] = []

    def run_mapping(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        graph = mapping_input.graph
        edges = np.asarray(graph.get_edges())[:, :2].astype(np.int64)
        n = int(graph.num_vertices())

        # init: cores from the clusterer (cluster id == core id)
        cluster_output = self.clusterer.cluster(mapping_input)
        init_core = np.asarray(cluster_output.cluster_assignment, dtype=np.int64)
        # init: local indices from the address assigner
        init_idx = np.asarray(self.address_assigner.assign_addresses(mapping_input, cluster_output),
                              dtype=np.int64)

        solver = FeedbackLoopSolver(mapping_input.hw_config, edges[:, 0], edges[:, 1], n,
                                    self.config, mapping_input.seed)
        core, idx, k, history = solver.solve(init_core, init_idx)
        self.last_history = history

        assignment = MosaicAssignment(
            hw=mapping_input.hw_config,
            neuron_core_pre_assignment=core,
            neuron_idx_pre_assignment=idx,
            neuron_slice_assignment=k,
        )
        return MosaicNetworkMappingState(assignment=assignment, mapping_input=mapping_input)