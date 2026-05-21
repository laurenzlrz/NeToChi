import numpy as np

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMosaicMappingState


def validate_mosaic_mapping(
        config: MosaicHardwareConfig,
        state: BaseMosaicMappingState
) -> bool:
    """
    Validates that the mapping output fulfills the hardware constraints defined in MosaicHardwareConfig.

    Raises:
        ValueError: If any hardware constraint or shape consistency check fails.
    """

    # ---------------------------------------------------------
    # Type Checks
    # ---------------------------------------------------------

    # 1. Validate 'neuron_core_idxs_assignment' (Must be a 1D Integer Array)
    if not isinstance(state.neuron_core_idxs_assignment, np.ndarray):
        raise TypeError("neuron_core_idxs_assignment must be a numpy ndarray")
    if not np.issubdtype(state.neuron_core_idxs_assignment.dtype, np.integer):
        raise TypeError(
            f"neuron_core_idxs_assignment must contain integers, got {state.neuron_core_idxs_assignment.dtype}")
    if state.neuron_core_idxs_assignment.ndim != 1:
        raise ValueError(f"neuron_core_idxs_assignment must be 1D, got {state.neuron_core_idxs_assignment.ndim}D")

    # 2. Validate 'neuron_local_idxs_assignment' (Must be a 1D Integer Array)
    if not isinstance(state.neuron_local_idxs_assignment, np.ndarray):
        raise TypeError("neuron_local_idxs_assignment must be a numpy ndarray")
    if not np.issubdtype(state.neuron_local_idxs_assignment.dtype, np.integer):
        raise TypeError(
            f"neuron_local_idxs_assignment must contain integers, got {state.neuron_local_idxs_assignment.dtype}")
    if state.neuron_local_idxs_assignment.ndim != 1:
        raise ValueError(f"neuron_local_idxs_assignment must be 1D, got {state.neuron_local_idxs_assignment.ndim}D")

    # 3. Validate 'neuron_slice_assignments' (Must be a 2D Integer Matrix)
    if not isinstance(state.neuron_slice_assignments, np.ndarray):
        raise TypeError("neuron_slice_assignments must be a numpy ndarray")
    if not np.issubdtype(state.neuron_slice_assignments.dtype, np.integer):
        raise TypeError(f"neuron_slice_assignments must contain integers, got {state.neuron_slice_assignments.dtype}")
    if state.neuron_slice_assignments.ndim != 2:
        raise ValueError(f"neuron_slice_assignments must be a 2D matrix, got {state.neuron_slice_assignments.ndim}D")


    # ---------------------------------------------------------
    # Shape Consistency Checks
    # ---------------------------------------------------------

    # 1. Cross-attribute Alignment Check (Ensure neuron_idx dimensions match perfectly)
    num_neurons_core = state.neuron_core_idxs_assignment.shape[0]
    num_neurons_local = state.neuron_local_idxs_assignment.shape[0]
    num_neurons_slice = state.neuron_slice_assignments.shape[0]

    if not (num_neurons_core == num_neurons_local == num_neurons_slice):
        raise ValueError(
            f"Dimension mismatch along the neuron axis! "
            f"Core assignments has {num_neurons_core} neurons, "
            f"Local assignments has {num_neurons_local} neurons, "
            f"Slice assignments has {num_neurons_slice} neurons."
        )

    # 2. Distances range from 0 to config.max_distance inclusive
    expected_slice_cols = config.max_distance + 1
    if state.neuron_slice_assignments.shape[1] != expected_slice_cols:
        raise ValueError(
            f"Slice assignment dimension mismatch: Expected {expected_slice_cols} "
            f"distance columns, got {state.neuron_slice_assignments.shape[1]}."
        )


    # ---------------------------------------------------------
    # Hardware Boundary Checks
    # ---------------------------------------------------------

    # 1. neuron core ids within [0, num_cores)
    if np.any((state.neuron_core_idxs_assignment < 0) | (state.neuron_core_idxs_assignment >= config.total_cores)):
        raise ValueError(
            f"Core assignment out of bounds. Must be in [0, {config.total_cores - 1}]."
        )

    # 2. local indices within [0, neurons_per_core)
    if np.any((state.neuron_local_idxs_assignment < 0) | (state.neuron_local_idxs_assignment >= config.neurons_per_core)):
        raise ValueError(
            f"Local neuron assignment out of bounds. Must be in [0, {config.neurons_per_core - 1}]."
        )

    # 3. slice indices within [0, max_slices_for_dist)
    for dist in range(expected_slice_cols):
        max_slices_for_dist = config.num_slices_at_distance(dist)
        slice_col = state.neuron_slice_assignments[:, dist]

        if np.any((slice_col < 0) | (slice_col >= max_slices_for_dist)):
            raise ValueError(
                f"Slice assignment out of bounds for distance {dist}. "
                f"Allowed slices: [0, {max_slices_for_dist - 1}]."
            )

    # ---------------------------------------------------------
    # Collision Detection (Uniqueness Check)
    # ---------------------------------------------------------
    # 1. (core, local_idx) only occupied by one neuron
    flat_hardware_indices = (state.neuron_core_idxs_assignment * config.neurons_per_core) + state.neuron_local_idxs_assignment
    unique_indices = np.unique(flat_hardware_indices)

    if unique_indices.size != state.mapping_input.graph.num_vertices():
        raise ValueError(
            "Hardware collision detected: Multiple logical neurons are mapped to the same physical core and local index."
        )

    return True