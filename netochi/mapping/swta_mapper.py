"""
Structure-aware mapper for soft Winner-Takes-All (sWTA) networks (see ``SwtaFactory``).

The mapping it returns is valid, so its inconsistency count is an upper bound on the
optimal number of inconsistencies for the input. The mapper uses the known sWTA structure
(E chain + global inhibition) to make that bound as tight as possible.

Key observations
----------------
1.  Slice-0 visibility. If every target listens to slice 0 at every distance, a source at
    local index x is seen by all targets at distance d <= band(x), where
    band(x) = max{d : x < end of slice 0 at distance d}. Low indices are visible far away,
    high indices only inside their own core. Under this policy the number of valid edges is
        sum_s  w(s, band(x_s)),   w(s, b) = #out-edges of s to targets at distance <= b,
    so, once cores are fixed, the index assignment decomposes into one independent linear
    assignment problem per core, which we solve exactly (Hungarian algorithm).

2.  Spread the global inhibition. I->E is the second-largest edge block (p_i_to_e * |I| * |E|).
    If the I neurons are split evenly over all cores (n / #cores per core) at the lowest local
    indices, every E target sees every I neuron: the local ones directly and the remote ones
    through slice 0 at every distance. With the I cluster on its own core instead, a target
    at distance d only sees n / 2^d of the I neurons, which loses most I->E edges.

3.  Homes + dump. Spreading I leaves n - q free slots per core (q = I neurons per core), so an
    E cluster no longer fits on one core. Every E cluster gets a home core holding n - q of its
    neurons, and the leftover q neurons ("surplus") go to cores without a home ("dump" cores),
    nearest first. Small surplus pieces are cheap: a piece of size a cut from its cluster costs
    at most 2 * a * (n - a) edges. Filling the cores contiguously instead creates large cuts.

After the construction, an optional hill climb swaps blocks of E neurons between cores.
Finally the exact optimal slice per (target, distance) is computed, which can only add valid
edges on top of the slice-0 value.
"""
import time
from typing import List, Optional, Tuple

import icontract
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from netochi.input_generator.interfaces import MosaicAssignment, MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState

# Description keys written by SwtaFactory
DESC_NUM_CLUSTERS = "num_clusters"
DESC_NEURONS_PER_CLUSTER = "neurons_per_cluster"
DESC_NUM_I_CLUSTERS = "num_i_clusters"

# Block sizes tried by the hill climb when exchanging neurons between two cores
REFINE_BLOCK_SIZES = (1, 2, 4, 8, 16, 32)


class SwtaMapperConfig(BaseModel):
    num_clusters: Optional[int] = Field(default=None, gt=0, description="Number of sWTA clusters. Read from the input descriptions if None.")
    neurons_per_cluster: Optional[int] = Field(default=None, gt=0, description="Neurons per cluster. Read from the input descriptions if None.")
    num_i_clusters: Optional[int] = Field(default=None, ge=0, description="Number of inhibitory clusters (the last clusters). Read from the input descriptions if None.")
    refine_time_s: float = Field(default=10.0, ge=0.0, description="Time budget of the block-swap hill climb (0 disables it).")
    refine_max_iters: int = Field(default=5000, ge=0, description="Maximal number of hill-climb moves.")

    def create(self) -> "SwtaMapper":
        return SwtaMapper(config=self)


class _LayoutEvaluator:
    """
    Holds a core assignment and evaluates it under the slice-0 policy, with the index assignment
    solved exactly per core. Moving neurons updates the out-degree-per-core table incrementally.
    """

    def __init__(self, hw: MosaicHardwareConfig, src: npt.NDArray[np.int64], tgt: npt.NDArray[np.int64],
                 num_neurons: int, cores: npt.NDArray[np.int64]) -> None:
        self.hw = hw
        self.num_cores = hw.total_cores
        levels = hw.router_levels
        dist = np.array([[hw.core_distance(a, b) for b in range(self.num_cores)] for a in range(self.num_cores)])
        # band[x]: largest distance at which local index x lies in slice 0
        self.band = np.array([max([0] + [d for d in range(1, levels + 1) if x < hw.get_slice_bounds(d, 0)[1]])
                              for x in range(hw.neurons_per_core)], dtype=np.int64)
        # reach[c_src, c_tgt, b] = 1 iff a source on c_src in band b reaches targets on c_tgt
        self._reach = (dist[:, :, None] <= np.arange(levels + 1)[None, None, :]).astype(np.int64)

        order = np.argsort(tgt, kind="stable")
        self._in_ptr = np.searchsorted(tgt[order], np.arange(num_neurons + 1))
        self._in_src = src[order]

        self.cores = cores.astype(np.int64).copy()
        # out_per_core[s, c]: number of out-edges of s into core c
        self._out_per_core = np.zeros((num_neurons, self.num_cores), dtype=np.int64)
        np.add.at(self._out_per_core, (src, self.cores[tgt]), 1)

    def value(self) -> Tuple[int, npt.NDArray[np.int64]]:
        """Returns (valid edges under the slice-0 policy, optimal local indices)."""
        w = np.einsum("sc,scb->sb", self._out_per_core, self._reach[self.cores])
        indices = np.zeros(len(self.cores), dtype=np.int64)
        total = 0
        for core in range(self.num_cores):
            members = np.flatnonzero(self.cores == core)
            if members.size == 0:
                continue
            gain = w[members][:, self.band]  # rows: neurons, columns: local slots
            rows, slots = linear_sum_assignment(gain, maximize=True)
            total += int(gain[rows, slots].sum())
            indices[members[rows]] = slots
        return total, indices

    def move(self, neurons: npt.NDArray[np.int64], new_cores: npt.NDArray[np.int64]) -> None:
        for v, core in zip(neurons, new_cores):
            sources = self._in_src[self._in_ptr[v]:self._in_ptr[v + 1]]
            np.add.at(self._out_per_core[:, self.cores[v]], sources, -1)
            np.add.at(self._out_per_core[:, core], sources, 1)
            self.cores[v] = core


def optimal_slices(hw: MosaicHardwareConfig, src: npt.NDArray[np.int64], tgt: npt.NDArray[np.int64],
                 cores: npt.NDArray[np.int64], indices: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """For every (target, distance): the slice containing the most in-neighbours (exact)."""
    num_neurons = cores.size
    levels = hw.router_levels
    slices = np.zeros((num_neurons, levels + 1), dtype=np.int64)
    if src.size == 0:
        return slices
    dist = np.array([[hw.core_distance(a, b) for b in range(hw.total_cores)] for a in range(hw.total_cores)])
    edge_dist = dist[cores[tgt], cores[src]]
    for d in range(1, levels + 1):
        num_slices = hw.num_slices_at_distance(d)
        slice_of = np.array([hw.get_slice_idx(d, x) for x in range(hw.neurons_per_core)], dtype=np.int64)
        at_d = edge_dist == d
        counts = np.bincount(tgt[at_d] * num_slices + slice_of[indices[src[at_d]]],
                             minlength=num_neurons * num_slices).reshape(num_neurons, num_slices)
        slices[:, d] = counts.argmax(axis=1)
    return slices


class SwtaMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper for sWTA networks: spreads the inhibitory clusters evenly over all cores, gives every
    excitatory cluster a home core, sends the surplus to the nearest free slots, and solves the
    index and slice assignment exactly. See the module docstring for the reasoning.
    """

    @icontract.require(lambda config: isinstance(config, SwtaMapperConfig))
    def __init__(self, config: SwtaMapperConfig) -> None:
        self.config = config

    def get_name(self) -> str:
        return "SwtaMapper"

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def _structure(self, mapping_input: MosaicMappingInput) -> Tuple[npt.NDArray[np.int64], int, int]:
        """Returns (cluster label per neuron, number of E clusters, number of I clusters)."""
        desc = mapping_input.descriptions

        def get(value: Optional[int], key: str) -> int:
            if value is not None:
                return value
            if key not in desc:
                raise ValueError(f"SwtaMapper needs '{key}': set it in SwtaMapperConfig or in the input descriptions.")
            return int(desc[key])

        num_clusters = get(self.config.num_clusters, DESC_NUM_CLUSTERS)
        per_cluster = get(self.config.neurons_per_cluster, DESC_NEURONS_PER_CLUSTER)
        num_i = get(self.config.num_i_clusters, DESC_NUM_I_CLUSTERS)
        if num_clusters * per_cluster != mapping_input.num_neurons:
            raise ValueError(f"sWTA structure ({num_clusters} x {per_cluster}) does not match the "
                             f"{mapping_input.num_neurons} neurons of the graph.")
        if num_i > num_clusters:
            raise ValueError("More inhibitory clusters than clusters.")
        labels = np.arange(mapping_input.num_neurons, dtype=np.int64) // per_cluster
        return labels, num_clusters - num_i, num_i

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @staticmethod
    def _spread_inhibition(hw: MosaicHardwareConfig, i_neurons: npt.NDArray[np.int64],
                           cores: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Round-robin over the cores, so every I cluster is spread evenly. Returns free slots per core."""
        cores[i_neurons] = np.arange(i_neurons.size) % hw.total_cores
        used = np.bincount(cores[i_neurons], minlength=hw.total_cores)
        return hw.neurons_per_core - used

    @staticmethod
    def _contiguous_layout(e_neurons: npt.NDArray[np.int64], free: npt.NDArray[np.int64],
                           cores: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Fills the E chain into the cores in order."""
        cores = cores.copy()
        cores[e_neurons] = np.repeat(np.arange(free.size), free)[:e_neurons.size]
        return cores

    @staticmethod
    def _home_dump_layout(hw: MosaicHardwareConfig, e_clusters: List[npt.NDArray[np.int64]],
                          free: npt.NDArray[np.int64], cores: npt.NDArray[np.int64],
                          dumps: List[int]) -> Optional[npt.NDArray[np.int64]]:
        """
        E clusters get the non-dump cores as homes, in chain order. Each home takes as much of its
        cluster as fits; the surplus goes to the nearest cores with free slots.
        """
        homes = [k for k in range(hw.total_cores) if k not in dumps]
        if len(e_clusters) > len(homes):
            return None
        cores = cores.copy()
        free = free.copy()
        surplus: List[Tuple[int, npt.NDArray[np.int64]]] = []
        for home, members in zip(homes, e_clusters):
            take = min(int(free[home]), members.size)
            cores[members[:take]] = home
            free[home] -= take
            if take < members.size:
                surplus.append((home, members[take:]))
        if sum(rest.size for _, rest in surplus) > free.sum():
            return None
        for home, rest in surplus:
            by_distance = sorted(range(hw.total_cores), key=lambda k: (hw.core_distance(home, k), k))
            pos = 0
            for core in by_distance:
                if pos == rest.size:
                    break
                take = min(int(free[core]), rest.size - pos)
                cores[rest[pos:pos + take]] = core
                free[core] -= take
                pos += take
        return cores

    def _candidate_layouts(self, hw: MosaicHardwareConfig, labels: npt.NDArray[np.int64],
                           num_e: int) -> List[npt.NDArray[np.int64]]:
        num_cores = hw.total_cores
        cores = np.full(labels.size, -1, dtype=np.int64)
        free = self._spread_inhibition(hw, np.flatnonzero(labels >= num_e), cores)
        e_neurons = np.flatnonzero(labels < num_e)
        e_clusters = [np.flatnonzero(labels == k) for k in range(num_e)]

        candidates = [self._contiguous_layout(e_neurons, free, cores)]
        num_dumps = num_cores - num_e
        if num_dumps >= 0:
            # Dump cores evenly spaced over the core order; try every offset.
            spacing = num_cores // num_dumps if num_dumps > 0 else 1
            for offset in range(spacing):
                dumps = [(offset + i * spacing) % num_cores for i in range(num_dumps)]
                layout = self._home_dump_layout(hw, e_clusters, free, cores, dumps)
                if layout is not None:
                    candidates.append(layout)
        return candidates

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------
    def _refine(self, evaluator: _LayoutEvaluator, labels: npt.NDArray[np.int64], num_e: int,
                value: int, rng: np.random.Generator) -> int:
        """Hill climb: exchange equally sized blocks of two E clusters between two cores."""
        if self.config.refine_time_s <= 0 or self.config.refine_max_iters == 0 or evaluator.num_cores < 2:
            return value
        deadline = time.monotonic() + self.config.refine_time_s
        is_e = labels < num_e
        for _ in range(self.config.refine_max_iters):
            if time.monotonic() > deadline:
                break
            core_a, core_b = rng.choice(evaluator.num_cores, size=2, replace=False)
            on_a = np.flatnonzero((evaluator.cores == core_a) & is_e)
            on_b = np.flatnonzero((evaluator.cores == core_b) & is_e)
            if on_a.size == 0 or on_b.size == 0:
                continue
            cluster_a = rng.choice(np.unique(labels[on_a]))
            cluster_b = rng.choice(np.unique(labels[on_b]))
            if cluster_a == cluster_b:
                continue
            block_a = on_a[labels[on_a] == cluster_a]
            block_b = on_b[labels[on_b] == cluster_b]
            k = min(int(rng.choice(REFINE_BLOCK_SIZES)), block_a.size, block_b.size)
            moved = np.concatenate([rng.choice(block_a, k, replace=False), rng.choice(block_b, k, replace=False)])
            to_new = np.array([core_b] * k + [core_a] * k, dtype=np.int64)
            to_old = np.array([core_a] * k + [core_b] * k, dtype=np.int64)

            evaluator.move(moved, to_new)
            new_value, _ = evaluator.value()
            if new_value >= value:
                value = new_value
            else:
                evaluator.move(moved, to_old)
        return value

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def run_mapping(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        hw = mapping_input.hw_config
        labels, num_e, _ = self._structure(mapping_input)
        edges = mapping_input.graph.get_edges()
        src = edges[:, 0].astype(np.int64)
        tgt = edges[:, 1].astype(np.int64)

        # 1. Construction: keep the best candidate layout
        best: Optional[Tuple[int, npt.NDArray[np.int64]]] = None
        for layout in self._candidate_layouts(hw, labels, num_e):
            value, _ = _LayoutEvaluator(hw, src, tgt, labels.size, layout).value()
            if best is None or value > best[0]:
                best = (value, layout)
        assert best is not None

        # 2. Refinement of the core assignment
        evaluator = _LayoutEvaluator(hw, src, tgt, labels.size, best[1])
        self._refine(evaluator, labels, num_e, best[0], np.random.default_rng(mapping_input.seed))

        # 3. Exact index assignment (slice-0 policy), then exact slice assignment
        _, indices = evaluator.value()
        cores = evaluator.cores
        slices = optimal_slices(hw, src, tgt, cores, indices)

        assignment = MosaicAssignment(
            hw=hw,
            neuron_core_pre_assignment=cores.astype(np.int64),
            neuron_idx_pre_assignment=indices.astype(np.int64),
            neuron_slice_assignment=slices,
        )
        return MosaicNetworkMappingState(mapping_input=mapping_input, assignment=assignment)
