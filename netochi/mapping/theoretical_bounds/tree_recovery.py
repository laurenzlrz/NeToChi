"""
Tree recovery helpers (RESULT_v2_readable.md, Sections 6.1 and 6.7).

Both trees the algorithm needs - the router tree over the recovered core blocks and the address
tree over the recovered prefix classes - are complete binary trees whose leaves are recovered by
recursive balanced bisection of a similarity matrix:

  - router tree (6.1):   similarity between blocks a, b from the block statistic ("total" or "median")
  - address tree (6.7):  similarity between classes j1, j2 at level d is Psi_d(j1, j2)

Bisection rule: every item r of the current set proposes the split "r together with the
|set|/2 - 1 items most similar to r" (the own-row rule of Corollary 24 / the reference-block rule
of 6.1); the proposal with the largest total within-half similarity is taken. On exact data every
proposal is the true split. The leaf order returned by the recursion is the binary label of each
leaf, most significant bit = root split, which matches MosaicHardwareConfig.core_distance and the
dyadic slice layout (top d bits of the local address).
"""
from __future__ import annotations

from typing import Callable, List, Literal, Sequence, Tuple

import icontract
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BlockStatistic = Literal["total", "median"]


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


@icontract.require(lambda sym_weights: sym_weights.ndim == 2 and sym_weights.shape[0] == sym_weights.shape[1])
@icontract.require(lambda sym_weights, labels: labels.shape == (sym_weights.shape[0],))
@icontract.ensure(lambda result, num_blocks: result.shape == (num_blocks, num_blocks))
def block_similarity(
    sym_weights: FloatArray,
    labels: IntArray,
    num_blocks: int,
    statistic: BlockStatistic,
) -> FloatArray:
    """
    Similarity between recovered blocks from the per-vertex counts W(u, b) = sum_{x in b} W_ux.

      "total":  S(a, b) = sum_{u in a} W(u, b)        (Section 6.1, W(a, b))
      "median": S(a, b) = median_{u in a} W(u, b)     (robust to a few corrupted vertices)

    The result is symmetrised ((S + S^T) / 2); the diagonal is set to zero (unused).
    """
    onehot: FloatArray = np.zeros((labels.shape[0], num_blocks), dtype=np.float64)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    per_vertex: FloatArray = sym_weights @ onehot                # W(u, b), shape (N, num_blocks)

    sim: FloatArray = np.zeros((num_blocks, num_blocks), dtype=np.float64)
    for a in range(num_blocks):
        rows = per_vertex[labels == a]
        if rows.shape[0] == 0:
            continue
        if statistic == "total":
            sim[a] = rows.sum(axis=0)
        else:
            sim[a] = np.median(rows, axis=0)
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 0.0)
    return sim


@icontract.require(lambda items: len(items) >= 2 and len(items) % 2 == 0)
@icontract.ensure(lambda result, items: sorted(result[0] + result[1]) == sorted(items))
@icontract.ensure(lambda result: len(result[0]) == len(result[1]))
def balanced_bisection(items: Sequence[int], similarity: FloatArray) -> Tuple[List[int], List[int]]:
    """
    Split `items` into two equal halves of high within-half similarity (rule: module docstring).
    The half containing the smallest item is returned first (fixes the tree automorphism).
    """
    item_arr = np.asarray(sorted(items), dtype=np.int64)
    half = item_arr.shape[0] // 2
    sub: FloatArray = similarity[np.ix_(item_arr, item_arr)].astype(np.float64)
    np.fill_diagonal(sub, 0.0)

    best_score = -np.inf
    best_mask = np.zeros(item_arr.shape[0], dtype=bool)
    for r in range(item_arr.shape[0]):
        # most similar first, ties by position (deterministic); r itself is always included
        order = sorted((i for i in range(item_arr.shape[0]) if i != r), key=lambda i: (-sub[r, i], i))
        mask = np.zeros(item_arr.shape[0], dtype=bool)
        mask[r] = True
        mask[order[: half - 1]] = True
        score = float(sub[np.ix_(mask, mask)].sum() + sub[np.ix_(~mask, ~mask)].sum())
        if score > best_score:
            best_score, best_mask = score, mask

    first = [int(i) for i in item_arr[best_mask]]
    second = [int(i) for i in item_arr[~best_mask]]
    if first[0] > second[0]:
        first, second = second, first
    return first, second


@icontract.require(lambda items: _is_power_of_two(len(items)))
@icontract.ensure(lambda result, items: sorted(result) == sorted(items))
def recursive_balanced_bisection(
    items: Sequence[int],
    similarity_at_level: Callable[[int], FloatArray],
    level: int = 1,
) -> List[int]:
    """
    Order the leaves of a complete binary tree over `items` by recursive balanced bisection.

    similarity_at_level(d) gives the similarity used to split a subtree at depth d (root: d = 1).
    The position of an item in the returned list is its binary leaf label (MSB = root split).
    """
    if len(items) == 1:
        return [int(items[0])]
    first, second = balanced_bisection(items, similarity_at_level(level))
    return (recursive_balanced_bisection(first, similarity_at_level, level + 1)
            + recursive_balanced_bisection(second, similarity_at_level, level + 1))


@icontract.require(lambda adjacency, core, class_labels:
                   adjacency.shape == (core.shape[0], core.shape[0]) and class_labels.shape == core.shape)
@icontract.ensure(lambda result, num_classes, max_distance: result.shape == (max_distance + 1, num_classes, num_classes))
def class_cooccurrence_by_distance(
    adjacency: FloatArray,
    core: IntArray,
    class_labels: IntArray,
    reference_cores: Sequence[int],
    core_distance: IntArray,
    num_classes: int,
    max_distance: int,
) -> FloatArray:
    """
    Psi_d(j1, j2) = sum_{a in reference_cores} sum_{t : coredist(a, core(t)) = d} N^a_{j1}(t) N^a_{j2}(t)

    where N^a_j(t) counts the class-j sources in core a with an edge to t (Section 6.7). The proof
    uses a single reference core a; summing over all reference cores of the half only adds signal.

    adjacency[u, t] = 1 iff edge u -> t. class_labels must be defined (>= 0) on reference cores.
    Returns psi with psi[d] = Psi_d for d = 1..max_distance (psi[0] is unused and zero).
    """
    psi: FloatArray = np.zeros((max_distance + 1, num_classes, num_classes), dtype=np.float64)
    for a in reference_cores:
        sources = np.flatnonzero(core == a)
        if sources.size == 0:
            continue
        onehot: FloatArray = np.zeros((num_classes, sources.size), dtype=np.float64)
        onehot[class_labels[sources], np.arange(sources.size)] = 1.0
        dist_to_core = core_distance[a][core]                  # coredist(a, core(t)) per vertex t
        for d in range(1, max_distance + 1):
            targets = np.flatnonzero(dist_to_core == d)
            if targets.size == 0:
                continue
            counts: FloatArray = onehot @ adjacency[np.ix_(sources, targets)]   # N^a_j(t)
            psi[d] += counts @ counts.T
    for d in range(max_distance + 1):
        np.fill_diagonal(psi[d], 0.0)
    return psi
