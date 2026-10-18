"""
Recovery of every neuron's cell = (core, top-h address prefix), RESULT_v2_readable.md, Chapters 2 and 6.

Notation: N neurons, C = 2^h cores (h router levels), n = 2^L neurons per core (the proof's M).
X is the directed adjacency (X[u, t] = 1 iff edge u -> t), W = X + X^T.

  1. Core stage       SDP max <W, Y>, block size n, rounded to C blocks of exactly n neurons.
  2. Router tree      block statistic ("total" = proof, "median" = robust variant), recursive
                      balanced bisection -> each block gets its leaf (= core id).
  3. Prefix stage     for each root half A (other half B): W'_uv = |O_u cap O_v|, O_u = out-neighbours
                      of u in B; SDP with block size |A| / 2^h = n / 2; rounded to 2^h classes with
                      exactly n / 2^h members per (core, class).
  4. Address blocks   recursive balanced bisection of the classes, level d split by Psi_d summed
                      over all reference cores of the half -> each class gets its h-bit prefix.
  5. Local address    prefix * (n / 2^h) + rank of the neuron within its (core, prefix) cell
                      (by neuron id: the low L - h bits are not identifiable, Section 1.4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import icontract
import numpy as np
import numpy.typing as npt

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.theoretical_bounds.sdp_partition import (
    SolverOption,
    round_to_balanced_blocks,
    solve_balanced_partition_sdp,
)
from netochi.mapping.theoretical_bounds.tree_recovery import (
    BlockStatistic,
    block_similarity,
    class_cooccurrence_by_distance,
    recursive_balanced_bisection,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CellRecoveryResult:
    """Output of recover_cells: per-neuron core id, local address and h-bit address prefix."""
    core: IntArray
    local_address: IntArray
    prefix: IntArray
    stage_seconds: Dict[str, float] = field(default_factory=dict)


def _cells_are_injective(result: CellRecoveryResult, neurons_per_core: int) -> bool:
    flat = result.core * neurons_per_core + result.local_address
    return bool(np.unique(flat).size == flat.size)


@icontract.require(lambda adjacency, hw: adjacency.shape == (hw.total_neurons, hw.total_neurons))
@icontract.ensure(lambda result, hw: _cells_are_injective(result, hw.neurons_per_core))
@icontract.ensure(lambda result, hw: bool(np.all(np.bincount(result.core, minlength=hw.total_cores) == hw.neurons_per_core)))
def recover_cells(
    adjacency: FloatArray,
    hw: MosaicHardwareConfig,
    rng: np.random.Generator,
    block_statistic: BlockStatistic = "median",
    solver: str = "SCS",
    solver_options: Optional[Mapping[str, SolverOption]] = None,
    rounding_iterations: int = 20,
) -> CellRecoveryResult:
    """
    Run steps 1-5 of the module docstring. The caller must have checked that the hardware is
    supported (binary routers, slice factor 2, n a power of two with n >= 2^h, |V| = C n).
    """
    num_cores = hw.total_cores
    per_core = hw.neurons_per_core
    levels = hw.router_levels
    num_classes = 2 ** levels
    cell_size = per_core // num_classes
    timings: Dict[str, float] = {}

    sym: FloatArray = adjacency + adjacency.T

    # --- 1. core stage ---------------------------------------------------------------------
    t0 = time.perf_counter()
    y_core = solve_balanced_partition_sdp(sym, per_core, solver=solver, solver_options=solver_options)
    blocks = round_to_balanced_blocks(
        y_core, num_cores, np.zeros(adjacency.shape[0], dtype=np.int64), per_core, rng, rounding_iterations
    )
    timings["core_stage"] = time.perf_counter() - t0

    # --- 2. router tree ----------------------------------------------------------------------
    t0 = time.perf_counter()
    sim = block_similarity(sym, blocks, num_cores, block_statistic)
    leaf_order = recursive_balanced_bisection(list(range(num_cores)), lambda _level: sim)
    core_of_block: IntArray = np.empty(num_cores, dtype=np.int64)
    core_of_block[np.asarray(leaf_order, dtype=np.int64)] = np.arange(num_cores, dtype=np.int64)
    core: IntArray = core_of_block[blocks]
    core_distance: IntArray = np.array(
        [[hw.core_distance(a, b) for b in range(num_cores)] for a in range(num_cores)], dtype=np.int64
    )
    timings["router_tree"] = time.perf_counter() - t0

    # --- 3. + 4. prefix stage and address blocks, per root half --------------------------------
    t0 = time.perf_counter()
    prefix: IntArray = np.full(adjacency.shape[0], -1, dtype=np.int64)
    half_cores = num_cores // 2
    halves = (list(range(half_cores)), list(range(half_cores, num_cores)))
    for own, other in (halves, halves[::-1]):
        in_own = np.flatnonzero(np.isin(core, own))
        in_other = np.flatnonzero(np.isin(core, other))
        out_to_other: FloatArray = adjacency[np.ix_(in_own, in_other)]
        cooccurrence: FloatArray = out_to_other @ out_to_other.T          # W'_uv = |O_u cap O_v|

        y_prefix = solve_balanced_partition_sdp(
            cooccurrence, in_own.size // num_classes, solver=solver, solver_options=solver_options
        )
        classes = round_to_balanced_blocks(y_prefix, num_classes, core[in_own], cell_size, rng, rounding_iterations)

        class_labels: IntArray = np.full(adjacency.shape[0], -1, dtype=np.int64)
        class_labels[in_own] = classes
        psi = class_cooccurrence_by_distance(
            adjacency, core, class_labels, own, core_distance, num_classes, levels
        )
        class_order = recursive_balanced_bisection(list(range(num_classes)), lambda level: psi[level])
        prefix_of_class: IntArray = np.empty(num_classes, dtype=np.int64)
        prefix_of_class[np.asarray(class_order, dtype=np.int64)] = np.arange(num_classes, dtype=np.int64)
        prefix[in_own] = prefix_of_class[classes]
    timings["prefix_stage"] = time.perf_counter() - t0

    # --- 5. local addresses ------------------------------------------------------------------
    local_address: IntArray = np.empty(adjacency.shape[0], dtype=np.int64)
    for c in range(num_cores):
        members = np.flatnonzero(core == c)
        members = members[np.lexsort((members, prefix[members]))]      # by prefix, then neuron id
        rank_in_cell = np.arange(members.size, dtype=np.int64) % cell_size
        local_address[members] = prefix[members] * cell_size + rank_in_cell

    return CellRecoveryResult(core=core, local_address=local_address, prefix=prefix, stage_seconds=timings)
