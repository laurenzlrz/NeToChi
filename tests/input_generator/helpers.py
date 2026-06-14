"""Shared helpers for input_generator tests."""

from __future__ import annotations

import graph_tool.all as gt
import numpy as np
import numpy.typing as npt

from netochi.input_generator.interfaces import MosaicAssignment, MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


def make_hw_config(
    *,
    nodes_per_router: int = 2,
    neurons_per_core: int = 8,
    router_levels: int = 2,
    slice_factor: int = 2,
) -> MosaicHardwareConfig:
    return MosaicHardwareConfig(
        nodes_per_router=nodes_per_router,
        neurons_per_core=neurons_per_core,
        router_levels=router_levels,
        slice_factor=slice_factor,
    )


def make_graph(num_vertices: int) -> gt.Graph:
    graph = gt.Graph(directed=True)
    graph.add_vertex(num_vertices)
    return graph


def max_slice_index(hw: MosaicHardwareConfig, level: int) -> int:
    """Largest valid slice index at a router level: slice_factor^level - 1."""
    return min(hw.slice_factor**level - 1, hw.total_neurons - 1)


def make_valid_assignment(hw: MosaicHardwareConfig) -> MosaicAssignment:
    total_neurons = hw.total_neurons
    return MosaicAssignment(
        neuron_core_pre_assignment=np.arange(total_neurons, dtype=np.int64) // hw.neurons_per_core,
        neuron_idx_pre_assignment=np.arange(total_neurons, dtype=np.int64) % hw.neurons_per_core,
        neuron_slice_assignment=np.zeros(
            (total_neurons, hw.router_levels + 1),
            dtype=np.int64,
        ),
    )


def assignment_with_core(
    assignment: MosaicAssignment,
    neuron_idx: int,
    core: int,
) -> MosaicAssignment:
    cores = assignment.neuron_core_pre_assignment.copy()
    cores[neuron_idx] = core
    return assignment.model_copy(update={"neuron_core_pre_assignment": cores})


def assignment_with_local_idx(
    assignment: MosaicAssignment,
    neuron_idx: int,
    local_idx: int,
) -> MosaicAssignment:
    indices = assignment.neuron_idx_pre_assignment.copy()
    indices[neuron_idx] = local_idx
    return assignment.model_copy(update={"neuron_idx_pre_assignment": indices})


def assignment_with_slice_value(
    assignment: MosaicAssignment,
    neuron_idx: int,
    level: int,
    value: int,
) -> MosaicAssignment:
    slices = assignment.neuron_slice_assignment.copy()
    slices[neuron_idx, level] = value
    return assignment.model_copy(update={"neuron_slice_assignment": slices})


def assignment_with_truncated_neurons(assignment: MosaicAssignment) -> MosaicAssignment:
    return MosaicAssignment(
        neuron_core_pre_assignment=assignment.neuron_core_pre_assignment[:-1],
        neuron_idx_pre_assignment=assignment.neuron_idx_pre_assignment[:-1],
        neuron_slice_assignment=assignment.neuron_slice_assignment[:-1],
    )


def assignment_with_fewer_slice_columns(assignment: MosaicAssignment) -> MosaicAssignment:
    return MosaicAssignment(
        neuron_core_pre_assignment=assignment.neuron_core_pre_assignment.copy(),
        neuron_idx_pre_assignment=assignment.neuron_idx_pre_assignment.copy(),
        neuron_slice_assignment=assignment.neuron_slice_assignment[:, :-1],
    )


def make_mapping_input(
    hw: MosaicHardwareConfig,
    assignment: MosaicAssignment | None = None,
) -> MosaicMappingInput[None]:
    return MosaicMappingInput(
        graph=make_graph(hw.total_neurons),
        descriptions={},
        hw_config=hw,
        assignment=assignment,
    )


def slice_bounds_cover_core(
    hw: MosaicHardwareConfig,
    distance: int,
) -> None:
    """Assert slice bounds at distance partition [0, neurons_per_core) without gaps."""
    slices = hw.num_slices_at_distance(distance)
    bounds = [hw.get_slice_bounds(distance, idx) for idx in range(slices)]

    assert bounds[0][0] == 0
    assert bounds[-1][1] == hw.neurons_per_core
    for (prev_start, prev_end), (start, end) in zip(bounds, bounds[1:]):
        assert prev_end == start
        assert start < end


def predecessors_in_core(graph: gt.Graph, target: int, core: int, neurons_per_core: int) -> set[int]:
    core_start = core * neurons_per_core
    core_end = core_start + neurons_per_core
    return {
        int(src)
        for src in graph.get_in_neighbors(target)
        if core_start <= int(src) < core_end
    }
