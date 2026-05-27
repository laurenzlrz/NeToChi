from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict, model_validator

from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, HWMappingInput, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# -----------------------------------------------------------------------------
# Base Mapping State Interfaces
# -----------------------------------------------------------------------------

class MappingState[ANY_MAPPING_INPUT: MappingInput](BaseModel):
    """Base class for all mapping results."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False, strict=True) # only for pydantic
    mapping_input: ANY_MAPPING_INPUT

    def init_random_assignments(self, seed: Optional[int] = None) -> None:
        """Abstract initialization method for assignments.
        basically constructor"""
        pass

class HWNetworkMappingState[ANY_MAPPING_INPUT: MappingInput](MappingState[ANY_MAPPING_INPUT]):
    """
    Base class for states that infer hardware. 
    Does not strictly require hardware parameters in input, but provides/infers them.
    """

    def _init_random_hw(self, seed: Optional[int] = None) -> None:
        """Randomly initialize hardware configuration."""
        pass

class NetworkAssignmentState[WITH_HW_INPUT: HWMappingInput](MappingState[WITH_HW_INPUT]):
    """
    State for hardware-aware partitioning. 
    Requires specific hardware parameters in the input (WITH_HW_INPUT).
    """
    pass

# -----------------------------------------------------------------------------
# Mosaic Specific Interfaces
# -----------------------------------------------------------------------------

class BaseMosaicMappingState[ANY_MAPPING_INPUT: MappingInput](MappingState[ANY_MAPPING_INPUT]):
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


class MosaicNetworkMappingState(BaseMosaicMappingState[MosaicMappingInput], NetworkAssignmentState[MosaicMappingInput]):
    """
    State for pure partitioning/assignment mappers. 
    Input MUST contain the hardware configuration (MosaicMappingInput).
    """

    @model_validator(mode="after")
    def validate_assignment_to_hw(self) -> 'MosaicNetworkMappingState':
        self.mapping_input.hw_config.verify_assignment(self.assignment)
        return self

    @property
    def hw(self) -> MosaicHardwareConfig:
        """Cheating method to access hardware config directly from the input, since it's guaranteed to be there."""
        return self.mapping_input.hw_config

    @classmethod
    def from_input(cls, mapping_input: MosaicMappingInput) -> 'MosaicNetworkMappingState':
        hw = mapping_input.hw_config
        N: int = mapping_input.graph.num_vertices()
        return cls(
            mapping_input=mapping_input,
            assignment=MosaicAssignment(
                neuron_core_pre_assignment=np.zeros(N, dtype=int),
                neuron_idx_pre_assignment=np.zeros(N, dtype=int),
                neuron_slice_assignment=np.zeros((N, hw.max_distance + 1), dtype=int)
            )
        )

    def init_random_assignments(self, seed: Optional[int] = None) -> None:
        """Randomly initialize assignments respecting given hardware capacities."""
        hw = self.hw
        N: int = self.mapping_input.graph.num_vertices()
        rng = np.random.default_rng(seed)
        
        slots = [(c, x) for c in range(hw.total_cores) for x in range(hw.neurons_per_core)]
        rng.shuffle(slots)
        
        for i in range(N):
            self.assignment.neuron_core_pre_assignment[i], self.assignment.neuron_idx_pre_assignment[i] = slots[i]
            
        for d in range(1, hw.max_distance + 1):
            n_sl: int = hw.num_slices_at_distance(d)
            self.assignment.neuron_slice_assignment[:, d] = rng.integers(0, n_sl, size=N)

        self.mapping_input.hw_config.verify_assignment(self.assignment)


class MosaicHWMappingState[ANY_MAPPING_INPUT: MappingInput](BaseMosaicMappingState[ANY_MAPPING_INPUT], HWNetworkMappingState[ANY_MAPPING_INPUT]):
    """
    State for joint inference mappers. 
    Input is purely the network (MappingInput), but output includes the optimized hardware.
    """
    hw_config_inferred: MosaicHardwareConfig = Field(description="Inferred hardware configuration parameters relevant to the mapping process.")

    @property
    def hw(self) -> MosaicHardwareConfig:
        return self.hw_config_inferred

    @classmethod
    def create_from_guess(cls, mapping_input: ANY_MAPPING_INPUT, initial_hw_guess: MosaicHardwareConfig) -> 'MosaicHWMappingState[ANY_MAPPING_INPUT]':
        """
        Creates an uninitialized state using an initial hardware guess to dimension the arrays.
        This guess is generated by the mapper and does NOT leak ground truth information.
        """
        N: int = mapping_input.graph.num_vertices()
        return cls(
            mapping_input=mapping_input,
            hw_config_inferred=initial_hw_guess,
            assignment=MosaicAssignment(
                neuron_core_pre_assignment=np.zeros(N, dtype=int),
                neuron_idx_pre_assignment=np.zeros(N, dtype=int),
                neuron_slice_assignment=np.zeros((N, initial_hw_guess.max_distance + 1), dtype=int)
            )
        )

    def init_random_assignments(self, seed: Optional[int] = None) -> None:
        """Randomly initialize assignments respecting the learned hardware capacities."""
        hw = self.hw
        N: int = self.mapping_input.graph.num_vertices()
        rng = np.random.default_rng(seed)
        
        slots = [(c, x) for c in range(hw.total_cores) for x in range(hw.neurons_per_core)]
        rng.shuffle(slots)

        for i in range(N):
            self.assignment.neuron_core_pre_assignment[i], self.assignment.neuron_idx_pre_assignment[i] = slots[i]

        for d in range(1, hw.max_distance + 1):
            n_sl: int = hw.num_slices_at_distance(d)
            self.assignment.neuron_slice_assignment[:, d] = rng.integers(0, n_sl, size=N)

        self.hw_config_inferred.verify_assignment(self.assignment)


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
