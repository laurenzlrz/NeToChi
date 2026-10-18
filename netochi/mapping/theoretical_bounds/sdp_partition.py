"""
Balanced-partition SDP and its rounding.

SDP (RESULT_v2_readable.md, Chapter 2, steps 1 and 3)
-----------------------------------------------------
    max <W, Y>  s.t.  Y PSD,  Y >= 0 (entrywise),  diag(Y) = 1,  Y 1 = m 1

where m is the block size. In the planted model the unique optimum is the 0/1 block matrix of
the true partition (Theorem 12 / Lemma 25); in practice the solver returns an approximate,
possibly non-integral Y.

Rounding (not part of the proof, which needs none at exact recovery)
--------------------------------------------------------------------
Balanced k-means on the rows of Y:
  1. Centres: farthest-point initialisation, the first row drawn from the seeded rng. If Y is
     the exact block matrix, this picks exactly one row per block.
  2. Assignment: every row is assigned to a centre by a min-cost assignment (squared Euclidean
     distance; each centre replicated `capacity` times) solved per group with
     scipy.optimize.linear_sum_assignment. This enforces the balance exactly: within each group,
     every block receives exactly `capacity` rows.
  3. Centres are recomputed as block means; steps 2-3 repeat until the labels are stable.
The procedure is deterministic given the rng.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Union

import cvxpy as cp
import icontract
from cvxpy.atoms.affine.binary_operators import multiply as cp_multiply
from cvxpy.atoms.affine.diag import diag as cp_diag
from cvxpy.atoms.affine.sum import sum as cp_sum
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from netochi.definitions.exceptions import MappingError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SolverOption = Union[int, float, bool, str]

_ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")


@icontract.require(lambda weights: weights.ndim == 2 and weights.shape[0] == weights.shape[1])
@icontract.require(lambda weights, block_size: 1 <= block_size <= weights.shape[0])
@icontract.ensure(lambda result, weights: result.shape == weights.shape)
def solve_balanced_partition_sdp(
    weights: FloatArray,
    block_size: int,
    solver: str = "SCS",
    solver_options: Optional[Mapping[str, SolverOption]] = None,
) -> FloatArray:
    """
    Solve max <W, Y> over {Y PSD, Y >= 0, diag(Y) = 1, Y 1 = block_size 1}.

    W is symmetrised and scaled to max |W| = 1 before solving (neither changes the maximisers).
    Returns the (approximate) optimal Y. Raises MappingError if the solver fails.
    """
    num = int(weights.shape[0])
    if block_size == num:       # a single block: the feasible set is {J}
        return np.ones((num, num), dtype=np.float64)
    if block_size == 1:         # singleton blocks: the feasible set is {I}
        return np.eye(num, dtype=np.float64)

    sym: FloatArray = 0.5 * (weights + weights.T)
    scale = float(np.max(np.abs(sym)))
    if scale > 0.0:
        sym = sym / scale

    y = cp.Variable((num, num), PSD=True)
    constraints = [
        y >= 0,
        cp_diag(y) == 1,
        cp_sum(y, axis=1) == block_size,
    ]
    problem = cp.Problem(cp.Maximize(cp_sum(cp_multiply(sym, y))), constraints)
    solve: Callable[..., object] = problem.solve     # cvxpy leaves Problem.solve unannotated
    try:
        solve(solver=solver, **dict(solver_options or {}))
    except cp.error.SolverError as exc:
        raise MappingError(f"Balanced-partition SDP failed with solver {solver}: {exc}") from exc

    if y.value is None or str(problem.status) not in _ACCEPTED_STATUSES:
        raise MappingError(
            f"Balanced-partition SDP did not reach an optimum (solver {solver}, status {problem.status})."
        )
    result: FloatArray = np.asarray(y.value, dtype=np.float64)
    symmetric: FloatArray = 0.5 * (result + result.T)
    return symmetric


def _squared_distances(points: FloatArray, centres: FloatArray) -> FloatArray:
    """Pairwise squared Euclidean distances, shape (len(points), len(centres))."""
    sq: FloatArray = (
        np.sum(points ** 2, axis=1)[:, None]
        - 2.0 * points @ centres.T
        + np.sum(centres ** 2, axis=1)[None, :]
    )
    return np.maximum(sq, 0.0)


def _farthest_point_centres(points: FloatArray, num_blocks: int, rng: np.random.Generator) -> FloatArray:
    """Farthest-point initialisation; the first centre is a row chosen by the rng."""
    chosen = [int(rng.integers(points.shape[0]))]
    min_dist = _squared_distances(points, points[chosen])[:, 0]
    while len(chosen) < num_blocks:
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, _squared_distances(points, points[[nxt]])[:, 0])
    centres: FloatArray = points[chosen].copy()
    return centres


def _balanced_assignment(cost: FloatArray, groups: IntArray, capacity: int) -> IntArray:
    """
    Assign every row to a block so that, within each group, each block gets exactly `capacity`
    rows, minimising the total cost (exact, via linear_sum_assignment per group).
    """
    num_blocks = cost.shape[1]
    labels: IntArray = np.full(cost.shape[0], -1, dtype=np.int64)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        expanded = np.repeat(cost[rows], capacity, axis=1)   # column k*capacity + i -> block k
        row_ind, col_ind = linear_sum_assignment(expanded)
        labels[rows[np.asarray(row_ind, dtype=np.int64)]] = np.asarray(col_ind, dtype=np.int64) // capacity
    if np.any(labels < 0) or num_blocks == 0:
        raise MappingError("Balanced assignment left rows unassigned.")
    return labels


@icontract.require(lambda embedding, groups: embedding.shape[0] == groups.shape[0])
@icontract.require(lambda num_blocks, capacity: num_blocks >= 1 and capacity >= 1)
@icontract.ensure(
    lambda result, groups, num_blocks, capacity: all(
        np.array_equal(np.bincount(result[groups == g], minlength=num_blocks), np.full(num_blocks, capacity))
        for g in np.unique(groups)
    ),
)
def round_to_balanced_blocks(
    embedding: FloatArray,
    num_blocks: int,
    groups: IntArray,
    capacity: int,
    rng: np.random.Generator,
    max_iterations: int = 20,
) -> IntArray:
    """
    Round the rows of an SDP solution (or any embedding) to `num_blocks` blocks such that every
    group contains exactly `capacity` rows of every block (balanced k-means, see module docstring).

    groups: group id per row; each group must contain exactly num_blocks * capacity rows.
    Returns the block label (0..num_blocks-1) per row.
    """
    for g in np.unique(groups):
        size = int(np.count_nonzero(groups == g))
        if size != num_blocks * capacity:
            raise MappingError(
                f"Group {g} has {size} rows, but balanced rounding needs exactly "
                f"{num_blocks} blocks x {capacity} = {num_blocks * capacity}."
            )

    centres = _farthest_point_centres(embedding, num_blocks, rng)
    labels: Optional[IntArray] = None
    for _ in range(max_iterations):
        new_labels = _balanced_assignment(_squared_distances(embedding, centres), groups, capacity)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centres = np.stack([embedding[labels == k].mean(axis=0) for k in range(num_blocks)])
    if labels is None:
        raise MappingError("Balanced rounding needs at least one iteration.")
    return labels
