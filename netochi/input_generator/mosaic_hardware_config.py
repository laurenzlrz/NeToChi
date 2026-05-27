from pydantic import BaseModel, Field, computed_field, ConfigDict, model_validator
import numpy as np

from interfaces import MosaicAssignment
from netochi.definitions.exceptions import InvalidConfigError, DimensionError, InvalidAssignmentError


class MosaicHardwareConfig(BaseModel):
    """Configuration for the target neuromorphic hardware."""
    model_config = ConfigDict(frozen=True, strict=True)

    nodes_per_router: int = Field(gt=0, description="Number of nodes connected to each router.")
    neurons_per_core: int = Field(gt=0, description="Number of neurons in each core.")
    router_levels: int = Field(ge=0, description="Number of levels in the router hierarchy.")
    slice_factor: int = Field(default=2, gt=0, description="Factor determining slice sizes for fan-in.")

    @model_validator(mode="after")
    def validate_config(self) -> "MosaicHardwareConfig":
        if self.slice_factor > self.neurons_per_core:
            raise InvalidConfigError("slice_factor cannot be greater than neurons_per_core.")
        return self

    @computed_field
    @property
    def total_cores(self) -> int:
        return self.nodes_per_router ** self.router_levels

    @computed_field
    @property
    def total_neurons(self) -> int:
        return self.total_cores * self.neurons_per_core

    @computed_field
    @property
    def max_distance(self) -> int:
        return self.router_levels

    def core_distance(self, core_a: int, core_b: int) -> int:
        """Calculate the hierarchical distance between two cores."""
        if core_a == core_b:
            return 0

        base = self.nodes_per_router
        levels = self.router_levels

        for level in range(levels - 1, -1, -1):
            digit_a = (core_a // (base ** level)) % base
            digit_b = (core_b // (base ** level)) % base
            if digit_a != digit_b:
                return level + 1

        return 0

    def num_slices_at_distance(self, distance: int) -> int:
        """Number of slices a core is partitioned into at a given distance."""
        return min(self.slice_factor ** distance, self.neurons_per_core)

    def get_slice_bounds(self, distance: int, slice_idx: int) -> tuple[int, int]:
        """Return the (start, end) local addresses for a given slice at a distance."""
        slices = self.num_slices_at_distance(distance)
        start = (slice_idx * self.neurons_per_core) // slices
        end = ((slice_idx + 1) * self.neurons_per_core) // slices
        return start, end

    def is_valid_connection(self, source_core: int, target_core: int, source_local_addr: int, target_slice_idx: int) -> bool:
        """Check if a connection satisfies Fan-In constraints."""
        dist = self.core_distance(target_core, source_core)
        start, end = self.get_slice_bounds(dist, target_slice_idx)
        return start <= source_local_addr < end

    def get_slice_idx(self, dist, src_local_address):
        for s_idx in range(self.num_slices_at_distance(dist)):
            start, end = self.get_slice_bounds(dist, s_idx)
            if start <= src_local_address < end:
                return s_idx
        return -1

    def verify_assignment(self, assignment: MosaicAssignment) -> None:

        # Validation against hardware
        if assignment.neuron_core_pre_assignment.size != self.total_neurons:
            raise DimensionError(
                f"Length of neuron_core_pre_assignment ({assignment.neuron_core_pre_assignment.size}) "
                f"must match total neurons in hardware config ({self.total_neurons})."
            )

        invalid_cores = (assignment.neuron_core_pre_assignment < 0) | (
                    assignment.neuron_core_pre_assignment >= self.total_cores)
        invalid_indices = (assignment.neuron_idx_pre_assignment < 0) | (
                    assignment.neuron_idx_pre_assignment >= self.neurons_per_core)

        if np.any(invalid_cores) or np.any(invalid_indices):
            failed_idx = np.where(invalid_cores | invalid_indices)[0][0]
            core_val = assignment.neuron_core_pre_assignment[failed_idx]
            local_val = assignment.neuron_idx_pre_assignment[failed_idx]

            raise InvalidAssignmentError(
                f"Neuron {failed_idx} assigned to core {core_val} "
                f"with local idx {local_val} exceeds hardware limits."
            )

        if assignment.neuron_slice_assignment.shape[1] != self.router_levels + 1:
            raise DimensionError(
                f"neuron_slice_assignment must have {self.router_levels + 1} columns "
                f"to match router levels in hardware config."
            )

        slice_indices = np.arange(self.router_levels + 1)
        max_allowed = np.minimum(self.slice_factor ** slice_indices - 1, self.total_neurons - 1)
        invalid_slices = (assignment.neuron_slice_assignment < 0) | (assignment.neuron_slice_assignment > max_allowed)

        if np.any(invalid_slices):
            failed_neuron_idxs, failed_slice_idxs = np.where(invalid_slices)

            failed_neuron = failed_neuron_idxs[0]
            failed_slice = failed_slice_idxs[0]
            invalid_val = assignment.neuron_slice_assignment[failed_neuron, failed_slice]
            allowed_limit = max_allowed[failed_slice]

            raise InvalidAssignmentError(
                f"Neuron {failed_neuron} at slice index {failed_slice} "
                f"has invalid assignment {invalid_val}. "
                f"Maximum allowed for this slice is {allowed_limit} "
                f"({self.slice_factor}^{failed_slice} - 1)."
            )