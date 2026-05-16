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
    cores = state.neuron_core_idxs_assignment
    locals_idx = state.neuron_local_idxs_assignment
    slices = state.neuron_slice_assignments

    num_mapped_neurons = cores.shape[0]

    # ---------------------------------------------------------
    # 1. Shape Consistency Checks
    # ---------------------------------------------------------
    if locals_idx.shape[0] != num_mapped_neurons:
        raise ValueError(
            f"Shape mismatch: Found {cores.shape[0]} core assignments but "
            f"{locals_idx.shape[0]} local assignments."
        )

    if slices.shape[0] != num_mapped_neurons:
        raise ValueError(
            f"Shape mismatch: Found {cores.shape[0]} core assignments but "
            f"{slices.shape[0]} slice assignments."
        )

    # Distances range from 0 to config.max_distance inclusive
    expected_slice_cols = config.max_distance + 1
    if slices.shape[1] != expected_slice_cols:
        raise ValueError(
            f"Slice assignment dimension mismatch: Expected {expected_slice_cols} "
            f"distance columns, got {slices.shape[1]}."
        )

    # ---------------------------------------------------------
    # 2. Hardware Boundary Checks
    # ---------------------------------------------------------
    if np.any((cores < 0) | (cores >= config.total_cores)):
        raise ValueError(
            f"Core assignment out of bounds. Must be in [0, {config.total_cores - 1}]."
        )

    if np.any((locals_idx < 0) | (locals_idx >= config.neurons_per_core)):
        raise ValueError(
            f"Local neuron assignment out of bounds. Must be in [0, {config.neurons_per_core - 1}]."
        )

    # ---------------------------------------------------------
    # 3. Slice Boundary Checks (Per Distance)
    # ---------------------------------------------------------
    for dist in range(expected_slice_cols):
        max_slices_for_dist = config.num_slices_at_distance(dist)
        slice_col = slices[:, dist]

        if np.any((slice_col < 0) | (slice_col >= max_slices_for_dist)):
            raise ValueError(
                f"Slice assignment out of bounds for distance {dist}. "
                f"Allowed slices: [0, {max_slices_for_dist - 1}]."
            )

    # ---------------------------------------------------------
    # 4. Collision Detection (Uniqueness Check)
    # ---------------------------------------------------------
    # Map (core, local_idx) pairs to a single flat index across the whole chip
    flat_hardware_indices = (cores * config.neurons_per_core) + locals_idx
    unique_indices = np.unique(flat_hardware_indices)

    if unique_indices.size != num_mapped_neurons:
        raise ValueError(
            "Hardware collision detected: Multiple logical neurons are mapped "
            "to the same physical core and local index."
        )

    return True