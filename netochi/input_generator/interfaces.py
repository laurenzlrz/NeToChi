from abc import ABC, abstractmethod
from dataclasses import KW_ONLY
from typing import Dict, Optional, Any, List

import graph_tool.all as gt
import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, model_validator, BaseModel, Field

from netochi.definitions.exceptions import DimensionError, InvalidAssignmentError
from netochi.definitions.constants import INEQUAL_ASSIGNMENT_OBJECTS, TOO_MANY_NEURONS
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# -----------------------------------------------------------------------------
# Base HW Config Dataclasses
# -----------------------------------------------------------------------------

class MappingInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    graph: gt.Graph = Field(description="The input graph to be mapped onto hardware.")
    descriptions: Dict[str, str] = Field(description="Metadata describing the graph and mapping context.")

class HWMappingInput[HW_CONFIG](MappingInput):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    hw_config: HW_CONFIG = Field(description="Hardware configuration parameters relevant to the mapping process.")


class MosaicAssignment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    hw: MosaicHardwareConfig = Field(description="Hardware configuration for validating the assignment.")

    neuron_core_pre_assignment: npt.NDArray[np.int64] = Field(description="1D array mapping each neuron to a core index.")
    neuron_idx_pre_assignment: npt.NDArray[np.int64] = Field(description="1D array mapping each neuron to a local index within its assigned core.")
    neuron_slice_assignment: npt.NDArray[np.int64] = Field(description="2D array where each row corresponds to a neuron and each column corresponds to a router level, indicating the fan-in slice assignment for that neuron at that level.")
    #TODO: neuron_target_assignment: npt.NDArray[np.int64]

    @classmethod
    def spread(cls, num_neurons: int, hw: MosaicHardwareConfig) -> "MosaicAssignment":
        """Factory method to create a zero-initialized assignment."""
        neuron_core = np.arange(num_neurons, dtype=np.int64) // hw.neurons_per_core
        neuron_idx = np.arange(num_neurons, dtype=np.int64) % hw.neurons_per_core
        return cls(
            hw=hw,
            neuron_core_pre_assignment=neuron_core,
            neuron_idx_pre_assignment=neuron_idx,
            neuron_slice_assignment=np.zeros((num_neurons, hw.router_levels + 1), dtype=np.int64)
        )


    @classmethod
    def random(cls, num_neurons: int, hw: MosaicHardwareConfig, seed: Optional[int]) -> "MosaicAssignment":
        """Factory method to create a zero-initialized assignment."""
        ass = cls(
            hw=hw,
            neuron_core_pre_assignment=np.zeros(num_neurons, dtype=np.int64),
            neuron_idx_pre_assignment=np.zeros(num_neurons, dtype=np.int64),
            neuron_slice_assignment=np.zeros((num_neurons, hw.router_levels + 1), dtype=np.int64)
        )
        ass._init_random_self(seed)
        return ass


    def _init_random_self(self, seed: Optional[int] = None) -> None:
        """In-place random initialization of the assignment arrays."""
        rng = np.random.default_rng(seed)
        num_neurons: int = self.neuron_core_pre_assignment.size

        # Core & Index allocation
        slots = [(c, x) for c in range(self.hw.total_cores) for x in range(self.hw.neurons_per_core)]
        rng.shuffle(slots)

        for i in range(num_neurons):
            self.neuron_core_pre_assignment[i], self.neuron_idx_pre_assignment[i] = slots[i]

        # Slice allocation
        for d in range(1, self.hw.max_distance + 1):
            n_sl: int = self.hw.num_slices_at_distance(d)
            self.neuron_slice_assignment[:, d] = rng.integers(0, n_sl, size=num_neurons)


    @model_validator(mode="after")
    def validate_shapes(self: "MosaicAssignment") -> "MosaicAssignment":
        if self.neuron_core_pre_assignment.ndim != 1:
            raise DimensionError("neuron_core_pre_assignment must be a 1D array.")
        if self.neuron_idx_pre_assignment.ndim != 1:
            raise DimensionError("neuron_idx_pre_assignment must be a 1D array.")
        if self.neuron_slice_assignment.ndim != 2:
            raise DimensionError("neuron_slice_assignment must be a 2D array.")

        if self.neuron_core_pre_assignment.size != self.neuron_idx_pre_assignment.size:
            raise DimensionError("neuron_core_pre_assignment and neuron_idx_pre_assignment must have the same length.")
        if self.neuron_core_pre_assignment.size != self.neuron_slice_assignment.shape[0]:
            raise DimensionError("First dimension of neuron_slice_assignment must match length of pre-assignments.")

        self.hw.verify_assignment(self)

        return self


class MosaicMappingInput(HWMappingInput[MosaicHardwareConfig]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)
    assignment: Optional[MosaicAssignment] = Field(default=None, description="Optional pre-assignment of neurons to cores and slices.")

    @model_validator(mode="after")
    def validate_pre_assignment(self) -> "MosaicMappingInput":
        if self.graph.num_vertices() > self.hw_config.total_neurons:
            raise DimensionError(TOO_MANY_NEURONS)

        if self.assignment is not None:
            if not self.assignment.hw is self.hw_config:  # Ensure the assignment's hardware config matches the input's hardware config
                raise InvalidAssignmentError(INEQUAL_ASSIGNMENT_OBJECTS)
            # TODO can be removed maybe
            self.hw_config.verify_assignment(self.assignment)
        return self


# -----------------------------------------------------------------------------
# Base Factory Interfaces
# -----------------------------------------------------------------------------

class BaseInputFactory[MAPPING_INPUT_CO: MappingInput](ABC):
    """Abstract base class for factories that generate MappingInputs."""

    @abstractmethod
    def generate(self) -> MAPPING_INPUT_CO:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

class HWBaseInputFactory[WITH_HW_INPUT_CO: HWMappingInput[Any]](BaseInputFactory[WITH_HW_INPUT_CO]
):
    """Abstract base class for factories that generate MappingInputs with corresponding hardware."""
    pass