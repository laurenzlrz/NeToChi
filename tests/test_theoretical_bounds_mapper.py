"""Tests for the SDP-based TheoreticalBoundsMapper and its helpers."""
import time

import networkx as nx
import numpy as np
import pytest

from netochi.definitions.exceptions import HardwareConstraintError, MappingError, MappingValidationError
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
from netochi.input_generator.noisy_mosaic_network_factory import NoisyMosaicNetworkConfig
from netochi.input_generator.utils import nx_to_gt
from netochi.mapping.theoretical_bounds.sdp_partition import round_to_balanced_blocks, solve_balanced_partition_sdp
from netochi.mapping.theoretical_bounds.tree_recovery import recursive_balanced_bisection
from netochi.mapping.theoretical_bounds_mapper import TheoreticalBoundsMapperConfig
from netochi.objectives.utils import compute_inconsistencies
from tests.utils_mapping_output_validation import validate_mosaic_mapping


def _in_edge_data(mapping_input: MosaicMappingInput) -> dict:
    graph = mapping_input.graph
    return {
        'N': graph.num_vertices(),
        'm': graph.num_edges(),
        'in_edges': [[int(src) for src in v.in_neighbors()] for v in graph.vertices()],
    }


def _same_partition(labels_a: np.ndarray, labels_b: np.ndarray) -> bool:
    """True iff the two labelings induce the same partition (labels may be permuted)."""
    pairs = set(zip(labels_a.tolist(), labels_b.tolist()))
    return len(pairs) == len(set(labels_a.tolist())) == len(set(labels_b.tolist()))


def _hw(levels: int, per_core: int, nodes_per_router: int = 2) -> MosaicHardwareConfig:
    return MosaicHardwareConfig(
        nodes_per_router=nodes_per_router,
        neurons_per_core=per_core,
        router_levels=levels,
        slice_factor=2,
    )


# --- End-to-end on the planted Mosaic model ---

@pytest.mark.parametrize("block_statistic", ["median", "total"])
@pytest.mark.parametrize("per_core", [16, 32])
def test_noise_free_mosaic_is_recovered_exactly(per_core, block_statistic):
    """
    h = 2 router levels, p = 0.9, no noise: the proof predicts exact recovery of every cell
    (core, top-h address prefix), hence a valid mapping with zero inconsistencies.
    """
    levels = 2
    hw = _hw(levels, per_core)
    mapping_input = MosaicNetworkConfig(hw_config=hw, probability=0.9, seed=1).create().generate()
    mapper = TheoreticalBoundsMapperConfig(block_statistic=block_statistic).create()

    start = time.perf_counter()
    state = mapper.run(mapping_input)
    elapsed = time.perf_counter() - start

    assert validate_mosaic_mapping(hw, state)
    assert compute_inconsistencies(state, _in_edge_data(mapping_input)) == 0

    planted = mapping_input.assignment
    assert _same_partition(state.c, planted.neuron_core_pre_assignment), "core partition not recovered"
    cell_size = per_core >> levels
    planted_cells = planted.neuron_core_pre_assignment * (2 ** levels) + planted.neuron_idx_pre_assignment // cell_size
    recovered_cells = state.c * (2 ** levels) + state.x // cell_size
    assert _same_partition(recovered_cells, planted_cells), "(core, prefix) cells not recovered"
    assert elapsed < 60.0


def test_noisy_mosaic_valid_and_not_worse_than_planted():
    """
    With insertion noise (epsilon = 0.05) the planted assignment violates exactly the noise edges.
    On this instance the mapper recovers the planted cells, so it is not worse than that bound.
    """
    hw = _hw(2, 16)
    factory = NoisyMosaicNetworkConfig(hw_config=hw, probability=0.9, seed=1, epsilon=0.05).create()
    mapping_input = factory.generate()

    state = TheoreticalBoundsMapperConfig().create().run(mapping_input)

    assert validate_mosaic_mapping(hw, state)
    assert compute_inconsistencies(state, _in_edge_data(mapping_input)) <= factory.num_noise_edges


# --- Unsupported hardware is rejected explicitly ---

def test_rejects_non_binary_routers():
    hw = _hw(levels=1, per_core=4, nodes_per_router=3)
    mapping_input = MosaicNetworkConfig(hw_config=hw, probability=0.9, seed=1).create().generate()
    with pytest.raises(HardwareConstraintError):
        TheoreticalBoundsMapperConfig().create().run(mapping_input)


def test_rejects_non_power_of_two_core_size():
    hw = _hw(levels=1, per_core=12)
    mapping_input = MosaicNetworkConfig(hw_config=hw, probability=0.9, seed=1).create().generate()
    with pytest.raises(HardwareConstraintError):
        TheoreticalBoundsMapperConfig().create().run(mapping_input)


def test_rejects_too_few_address_bits():
    hw = _hw(levels=2, per_core=2)    # 2 < 2^2: no room for a 2-bit address prefix
    mapping_input = MosaicNetworkConfig(hw_config=hw, probability=0.9, seed=1).create().generate()
    with pytest.raises(HardwareConstraintError):
        TheoreticalBoundsMapperConfig().create().run(mapping_input)


def test_rejects_graph_not_filling_hardware():
    hw = _hw(levels=1, per_core=8)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(hw.total_neurons - 1))
    graph.add_edge(0, 1)
    mapping_input = MosaicMappingInput(graph=nx_to_gt(graph), descriptions={}, hw_config=hw, seed=1)
    with pytest.raises(MappingValidationError):
        TheoreticalBoundsMapperConfig().create().run(mapping_input)


def test_rejects_instances_above_max_neurons():
    hw = _hw(levels=1, per_core=8)
    mapping_input = MosaicNetworkConfig(hw_config=hw, probability=0.9, seed=1).create().generate()
    with pytest.raises(MappingError):
        TheoreticalBoundsMapperConfig(max_neurons=8).create().run(mapping_input)


# --- Helpers ---

def test_sdp_and_rounding_recover_planted_blocks():
    """Four dense blocks with sparse cuts, vertices shuffled: SDP + balanced rounding recover them."""
    rng = np.random.default_rng(0)
    truth = rng.permutation(np.repeat(np.arange(4), 6))
    weights = np.where(truth[:, None] == truth[None, :], 1.0, 0.05)
    np.fill_diagonal(weights, 0.0)

    y = solve_balanced_partition_sdp(weights, block_size=6)
    labels = round_to_balanced_blocks(y, 4, np.zeros(truth.size, dtype=np.int64), 6, rng)

    assert np.array_equal(np.bincount(labels, minlength=4), np.full(4, 6))
    assert _same_partition(labels, truth)


def test_balanced_rounding_respects_per_group_capacity():
    rng = np.random.default_rng(0)
    embedding = rng.random((16, 3))
    groups = np.repeat(np.arange(2), 8).astype(np.int64)
    labels = round_to_balanced_blocks(embedding, 4, groups, 2, rng)
    for g in range(2):
        assert np.array_equal(np.bincount(labels[groups == g], minlength=4), np.full(4, 2))


def test_recursive_bisection_recovers_ultrametric_tree():
    """Leaves of a depth-3 tree with similarity 2^-coredist, shuffled: the recovered order is a tree automorphism."""
    hw = _hw(levels=3, per_core=8)
    num = hw.total_cores
    perm = np.random.default_rng(3).permutation(num)            # item i sits at true leaf perm[i]
    sim = np.array([[2.0 ** -hw.core_distance(int(perm[i]), int(perm[j])) for j in range(num)]
                    for i in range(num)])

    order = recursive_balanced_bisection(list(range(num)), lambda _level: sim)

    recovered_leaf = np.empty(num, dtype=np.int64)
    recovered_leaf[np.asarray(order)] = np.arange(num)
    for i in range(num):
        for j in range(num):
            assert hw.core_distance(int(recovered_leaf[i]), int(recovered_leaf[j])) == \
                hw.core_distance(int(perm[i]), int(perm[j]))
