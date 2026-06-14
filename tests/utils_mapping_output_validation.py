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

    # 1. Validate 'state.c' (Must be a 1D Integer Array)
    if not isinstance(state.c, np.ndarray):
        raise TypeError(f"state.c must be a numpy ndarray, got {type(state.c).__name__}")
    if not np.issubdtype(state.c.dtype, np.integer):
        raise TypeError(
            f"state.c must contain integers, got {state.c.dtype}")
    if state.c.ndim != 1:
        raise ValueError(f"state.c must be 1D, got {state.c.ndim}D")

    # 2. Validate 'state.x' (Must be a 1D Integer Array)
    if not isinstance(state.x, np.ndarray):
        raise TypeError(f"state.x must be a numpy ndarray, got {type(state.x).__name__}")
    if not np.issubdtype(state.x.dtype, np.integer):
        raise TypeError(
            f"state.x must contain integers, got {state.x.dtype}")
    if state.x.ndim != 1:
        raise ValueError(f"state.x must be 1D, got {state.x.ndim}D")

    # 3. Validate 'state.s' (Must be a 2D Integer Matrix)
    if not isinstance(state.s, np.ndarray):
        raise TypeError(f"state.s must be a numpy ndarray, got {type(state.s).__name__}")
    if not np.issubdtype(state.s.dtype, np.integer):
        raise TypeError(f"state.s must contain integers, got {state.s.dtype}")
    if state.s.ndim != 2:
        raise ValueError(f"state.s must be a 2D matrix, got {state.s.ndim}D")


    # ---------------------------------------------------------
    # Shape Consistency Checks
    # ---------------------------------------------------------

    # 1. Cross-attribute Alignment Check (Ensure neuron_idx dimensions match perfectly)
    num_neurons_core = state.c.shape[0]
    num_neurons_local = state.x.shape[0]
    num_neurons_slice = state.s.shape[0]

    if not (num_neurons_core == num_neurons_local == num_neurons_slice):
        raise ValueError(
            f"Dimension mismatch along the neuron axis! "
            f"Core assignments has {num_neurons_core} neurons, "
            f"Local assignments has {num_neurons_local} neurons, "
            f"Slice assignments has {num_neurons_slice} neurons."
        )

    # 2. Distances range from 0 to config.max_distance inclusive
    expected_slice_cols = config.max_distance + 1
    if state.s.shape[1] != expected_slice_cols:
        raise ValueError(
            f"Slice assignment dimension mismatch: Expected {expected_slice_cols} "
            f"distance columns, got {state.s.shape[1]}."
        )


    # ---------------------------------------------------------
    # Hardware Boundary Checks
    # ---------------------------------------------------------

    # 1. neuron core ids within [0, num_cores)
    if np.any((state.c < 0) | (state.c >= config.total_cores)):
        raise ValueError(
            f"Core assignment out of bounds. Must be in [0, {config.total_cores - 1}]."
        )

    # 2. local indices within [0, neurons_per_core)
    if np.any((state.x < 0) | (state.x >= config.neurons_per_core)):
        raise ValueError(
            f"Local neuron assignment out of bounds. Must be in [0, {config.neurons_per_core - 1}]."
        )

    # 3. slice indices within [0, max_slices_for_dist)
    for dist in range(expected_slice_cols):
        max_slices_for_dist = config.num_slices_at_distance(dist)
        slice_col = state.s[:, dist]

        if np.any((slice_col < 0) | (slice_col >= max_slices_for_dist)):
            raise ValueError(
                f"Slice assignment out of bounds for distance {dist}. "
                f"Allowed slices: [0, {max_slices_for_dist - 1}]."
            )

    # ---------------------------------------------------------
    # Collision Detection (Uniqueness Check)
    # ---------------------------------------------------------
    # 1. (core, local_idx) only occupied by one neuron
    flat_hardware_indices = (state.c * config.neurons_per_core) + state.x
    unique_indices = np.unique(flat_hardware_indices)

    if unique_indices.size != state.mapping_input.graph.num_vertices():
        raise ValueError(
            "Hardware collision detected: Multiple logical neurons are mapped to the same physical core and local index."
        )

    return True