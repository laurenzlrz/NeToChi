from pydantic import BaseModel, Field, computed_field, ConfigDict, model_validator

from netochi.definitions.exceptions import InvalidConfigError


class MosaicHardwareConfig(BaseModel):
    """Configuration for the target neuromorphic hardware."""
    model_config = ConfigDict(frozen=True, strict=True)

    nodes_per_router: int = Field(gt=0, description="Number of nodes connected to each router.")
    neurons_per_core: int = Field(gt=0, description="Number of neurons in each core.")
    router_levels: int = Field(ge=0, description="Number of levels in the router hierarchy.")
    slice_factor: int = Field(default=2, gt=0, description="Factor determining slice sizes for fan-in.")

    @model_validator(mode="after")
    def validate_config(self) -> "MosaicHardwareConfig[BaseModel]":
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
