from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr

from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, HWMappingInput, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig

#TODO: Refactor init assignment to Assignment class, Mosaic Random Init to MosaicAssignment

# -----------------------------------------------------------------------------
# Base Mapping State Interfaces
# -----------------------------------------------------------------------------

class MappingState[ANY_MAPPING_INPUT: MappingInput, HW_CONFIG: Any](BaseModel):
    """Base class for all mapping results."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False, strict=True, populate_by_name=True)
    mapping_input: ANY_MAPPING_INPUT = Field(alias="_mapping_input", frozen=True)

    @property
    @abstractmethod
    def hw_to_evaluate(self) -> HW_CONFIG:
        pass

class HWNetworkMappingState[ANY_MAPPING_INPUT: MappingInput, INFERRED_HW_CONFIG: Any](MappingState[ANY_MAPPING_INPUT, INFERRED_HW_CONFIG]):
    """
    Base class for states that infer hardware. 
    Does not strictly require hardware parameters in input, but provides/infers them.
    """
    inferred_hw: INFERRED_HW_CONFIG = Field(alias="_inferred_hw_config", frozen=False)

    @property
    def hw_to_evaluate(self) -> INFERRED_HW_CONFIG:
        """For HW-aware mappers, the hardware to evaluate is the inferred hardware."""
        return self.inferred_hw

class NetworkAssignmentState[WITH_HW_INPUT: HWMappingInput, GT_HW_CONFIG: Any](MappingState[WITH_HW_INPUT, GT_HW_CONFIG]):
    """
    State for hardware-aware partitioning.
    Requires specific hardware parameters in the input (WITH_HW_INPUT).
    """
    @property
    def gt_hw(self) -> GT_HW_CONFIG:
        """Convenience property to access hardware config directly from the input."""
        return self.mapping_input.hw_config

    @property
    def hw_to_evaluate(self) -> GT_HW_CONFIG:
        """For hardware-aware mappers, the hardware to evaluate is the ground truth hardware from the input."""
        return self.gt_hw

# -----------------------------------------------------------------------------
# Mosaic Specific Interfaces
# -----------------------------------------------------------------------------

class BaseMosaicMappingState[ANY_MAPPING_INPUT: MappingInput](MappingState[ANY_MAPPING_INPUT, MosaicHardwareConfig]):
    """
    Abstract base state for all Mosaic mappers.  Infers HW
    Contains the assignment arrays and uses HWNetworkMappingState to ensure mapping_input exists.
    """
    assignment: MosaicAssignment = Field(description="The current found assignment of neurons to cores and slices.")

    @property
    def c(self) -> npt.NDArray[np.int_]: return self.assignment.neuron_core_pre_assignment

    @property
    def x(self) -> npt.NDArray[np.int_]: return self.assignment.neuron_idx_pre_assignment
    
    @property
    def s(self) -> np.ndarray[tuple[Any, Any], np.dtype[np.int_]]: return self.assignment.neuron_slice_assignment


class MosaicNetworkMappingState(BaseMosaicMappingState[MosaicMappingInput], NetworkAssignmentState[MosaicMappingInput, MosaicHardwareConfig]):
    """
    State for pure partitioning/assignment mappers. 
    Input MUST contain the hardware configuration (MosaicMappingInput).
    """

    @model_validator(mode="after")
    def validate_assignment_to_hw(self) -> 'MosaicNetworkMappingState':
        self.mapping_input.hw_config.verify_assignment(self.assignment)
        return self


    @classmethod
    def from_input_zero(cls, mapping_input: MosaicMappingInput) -> 'MosaicNetworkMappingState':
        hw = mapping_input.hw_config
        return cls(
            _mapping_input=mapping_input,
            assignment=MosaicAssignment.spread(num_neurons=mapping_input.graph.num_vertices(), hw=hw)
        )

    @classmethod
    def from_input_random(cls, mapping_input: MosaicMappingInput, seed: Optional[int] = None) -> 'MosaicNetworkMappingState':
        hw = mapping_input.hw_config
        return cls(
            _mapping_input=mapping_input,
            assignment=MosaicAssignment.random(num_neurons=mapping_input.graph.num_vertices(), hw=hw, seed=seed)
        )


class MosaicHWMappingState[ANY_MAPPING_INPUT: MappingInput](BaseMosaicMappingState[ANY_MAPPING_INPUT], HWNetworkMappingState[ANY_MAPPING_INPUT, MosaicHardwareConfig]):
    """
    State for joint inference mappers. 
    Input is purely the network (MappingInput), but output includes the optimized hardware.
    """

    @classmethod
    def from_guess_zero(cls, mapping_input: ANY_MAPPING_INPUT, initial_hw_guess: MosaicHardwareConfig) -> 'MosaicHWMappingState[ANY_MAPPING_INPUT]':
        """
        Creates an uninitialized state using an initial hardware guess to dimension the arrays.
        This guess is generated by the mapper and does NOT leak ground truth information.
        """
        return cls(
            _mapping_input=mapping_input,
            _inferred_hw_config=initial_hw_guess,
            assignment=MosaicAssignment.spread(
                hw = initial_hw_guess,
                num_neurons=mapping_input.graph.num_vertices()
            )
        )

    @classmethod
    def from_guess_random(cls, mapping_input: ANY_MAPPING_INPUT, initial_hw_guess: MosaicHardwareConfig, seed: Optional[int]) -> 'MosaicHWMappingState[ANY_MAPPING_INPUT]':
        return cls(
            _mapping_input=mapping_input,
            _inferred_hw_config=initial_hw_guess,
            assignment=MosaicAssignment.random(num_neurons=mapping_input.graph.num_vertices(), hw=initial_hw_guess, seed=seed)
        )


# -----------------------------------------------------------------------------
# Mapper Interface
# -----------------------------------------------------------------------------

class BaseMapper[MAPPING_STATE, ANY_MAPPING_INPUT](ABC):
    """Structural interface for mapping algorithms."""
    
    def get_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(self, mapping_input: ANY_MAPPING_INPUT) -> MAPPING_STATE:
        pass
