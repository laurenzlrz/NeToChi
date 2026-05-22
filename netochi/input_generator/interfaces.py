from abc import ABC, abstractmethod
from dataclasses import KW_ONLY
from typing import Dict, Optional, Any

import graph_tool.all as gt
import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from netochi.definitions.exceptions import DimensionError, NotSetError, InvalidAssignmentError
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# -----------------------------------------------------------------------------
# Base HW Config Dataclasses
# -----------------------------------------------------------------------------

@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class MappingInput[PAYLOAD]:
    _: KW_ONLY
    graph: gt.Graph
    descriptions: Dict[str, str]
    payload: Optional[PAYLOAD] = None

@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class HWMappingInput[PAYLOAD, HW_CONFIG](MappingInput[PAYLOAD]):
    _: KW_ONLY
    hw_config: HW_CONFIG

@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class MosaicMappingInput[PAYLOAD](HWMappingInput[PAYLOAD, MosaicHardwareConfig]):
    _: KW_ONLY
    neuron_core_pre_assignment: Optional[npt.NDArray[np.int_]] = None
    neuron_idx_pre_assignment: Optional[npt.NDArray[np.int_]] = None

    @model_validator(mode="after")
    def verify_pre_assignment(self) -> "MosaicMappingInput[PAYLOAD]":
        if self.neuron_core_pre_assignment is None and self.neuron_idx_pre_assignment is None:
            return self

        if self.neuron_core_pre_assignment is None or self.neuron_idx_pre_assignment is None:
            raise NotSetError(
                "Both neuron_core_pre_assignment and neuron_idx_pre_assignment must be provided together.")

        if self.neuron_core_pre_assignment.ndim != 1:
            raise DimensionError("neuron_core_pre_assignment must be a 1D array.")
        if self.neuron_idx_pre_assignment.ndim != 1:
            raise DimensionError("neuron_idx_pre_assignment must be a 1D array.")

        # Validation against hardware
        if self.neuron_core_pre_assignment.size != self.hw_config.total_neurons:
            raise DimensionError(
                f"Length of neuron_core_pre_assignment ({self.neuron_core_pre_assignment.size}) "
                f"must match total neurons in hardware config ({self.hw_config.total_neurons})."
            )

        if self.neuron_idx_pre_assignment.size != self.hw_config.total_neurons:
            raise DimensionError(
                f"Length of neuron_idx_pre_assignment ({self.neuron_idx_pre_assignment.size}) "
                f"must match total neurons in hardware config ({self.hw_config.total_neurons})."
            )

        invalid_cores = (self.neuron_core_pre_assignment < 0) | (
                    self.neuron_core_pre_assignment >= self.hw_config.total_cores)
        invalid_indices = (self.neuron_idx_pre_assignment < 0) | (
                    self.neuron_idx_pre_assignment >= self.hw_config.neurons_per_core)
        if np.any(invalid_cores) or np.any(invalid_indices):
            failed_idx = np.where(invalid_cores | invalid_indices)[0][0]
            core_val = self.neuron_core_pre_assignment[failed_idx]
            local_val = self.neuron_idx_pre_assignment[failed_idx]

            raise InvalidAssignmentError(
                f"Neuron {failed_idx} assigned to core {core_val} "
                f"with local idx {local_val} exceeds hardware limits."
            )

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

class HWBaseInputFactory[WITH_HW_INPUT_CO: HWMappingInput[Any, Any]](
    ABC, BaseInputFactory[WITH_HW_INPUT_CO]
):
    """Abstract base class for factories that generate MappingInputs."""
    pass