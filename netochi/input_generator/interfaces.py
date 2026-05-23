from abc import ABC, abstractmethod
from dataclasses import KW_ONLY
from typing import Dict, Optional, Any

import graph_tool.all as gt
import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, model_validator, BaseModel

from netochi.definitions.exceptions import DimensionError, InvalidAssignmentError
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# -----------------------------------------------------------------------------
# Base HW Config Dataclasses
# -----------------------------------------------------------------------------

class MappingInput[PAYLOAD](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    graph: gt.Graph
    descriptions: Dict[str, str]
    payload: Optional[PAYLOAD] = None

class HWMappingInput[PAYLOAD, HW_CONFIG](MappingInput[PAYLOAD]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    hw_config: HW_CONFIG


class MosaicAssignment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    neuron_core_pre_assignment: npt.NDArray[np.int64]
    neuron_idx_pre_assignment: npt.NDArray[np.int64]
    neuron_slice_assignment: npt.NDArray[np.int64]
    #TODO: neuron_target_assignment: npt.NDArray[np.int64]

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

        return self


class MosaicMappingInput[PAYLOAD](HWMappingInput[PAYLOAD, MosaicHardwareConfig]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    assignment: Optional[MosaicAssignment] = None

    @model_validator(mode="after")
    def validate_pre_assignment(self) -> "MosaicMappingInput[PAYLOAD]":
        if self.assignment is not None:
            self.verify_assignment(self.assignment)
        return self

    def verify_assignment(self, assignment: MosaicAssignment) -> None:

        # Validation against hardware
        if assignment.neuron_core_pre_assignment.size != self.hw_config.total_neurons:
            raise DimensionError(
                f"Length of neuron_core_pre_assignment ({assignment.neuron_core_pre_assignment.size}) "
                f"must match total neurons in hardware config ({self.hw_config.total_neurons})."
            )

        invalid_cores = (assignment.neuron_core_pre_assignment < 0) | (
                    assignment.neuron_core_pre_assignment >= self.hw_config.total_cores)
        invalid_indices = (assignment.neuron_idx_pre_assignment < 0) | (
                    assignment.neuron_idx_pre_assignment >= self.hw_config.neurons_per_core)

        if np.any(invalid_cores) or np.any(invalid_indices):
            failed_idx = np.where(invalid_cores | invalid_indices)[0][0]
            core_val = assignment.neuron_core_pre_assignment[failed_idx]
            local_val = assignment.neuron_idx_pre_assignment[failed_idx]

            raise InvalidAssignmentError(
                f"Neuron {failed_idx} assigned to core {core_val} "
                f"with local idx {local_val} exceeds hardware limits."
            )

        if assignment.neuron_slice_assignment.shape[1] != self.hw_config.router_levels + 1:
            raise DimensionError(
                f"neuron_slice_assignment must have {self.hw_config.router_levels + 1} columns "
                f"to match router levels in hardware config."
            )

        slice_indices = np.arange(self.hw_config.router_levels + 1)
        max_allowed = np.minimum(self.hw_config.slice_factor ** slice_indices - 1, self.hw_config.total_neurons - 1)
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
                f"({self.hw_config.slice_factor}^{failed_slice} - 1)."
            )



# -----------------------------------------------------------------------------
# Base Factory Interfaces
# -----------------------------------------------------------------------------

class BaseInputFactory[MAPPING_INPUT_CO: MappingInput[Any]](ABC):
    """Abstract base class for factories that generate MappingInputs."""

    @abstractmethod
    def generate(self) -> MAPPING_INPUT_CO:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

class HWBaseInputFactory[WITH_HW_INPUT_CO: HWMappingInput[Any, Any]](BaseInputFactory[WITH_HW_INPUT_CO]
):
    """Abstract base class for factories that generate MappingInputs."""
    pass