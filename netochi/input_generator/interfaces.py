from abc import ABC, abstractmethod
from dataclasses import KW_ONLY
from typing import Dict, Optional, Any

import graph_tool.all as gt
import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, model_validator, BaseModel, Field

from netochi.definitions.exceptions import DimensionError, InvalidAssignmentError
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

    neuron_core_pre_assignment: npt.NDArray[np.int64] = Field(description="1D array mapping each neuron to a core index.")
    neuron_idx_pre_assignment: npt.NDArray[np.int64] = Field(description="1D array mapping each neuron to a local index within its assigned core.")
    neuron_slice_assignment: npt.NDArray[np.int64] = Field(description="2D array where each row corresponds to a neuron and each column corresponds to a router level, indicating the fan-in slice assignment for that neuron at that level.")
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


class MosaicMappingInput(HWMappingInput[MosaicHardwareConfig]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    assignment: Optional[MosaicAssignment] = Field(default=None, description="Optional pre-assignment of neurons to cores and slices.")

    @model_validator(mode="after")
    def validate_pre_assignment(self) -> "MosaicMappingInput":
        if self.assignment is not None:
            self.hw_config.verify_assignment(self.assignment)
        return self


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