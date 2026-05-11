from typing import Dict, Any, TypeVar, Generic, Optional
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator
from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, WITH_HW_INPUT
from netochi.mapping.constants import (
    CORE_ASSIGNMENT_NOT_1D,
    CORE_ASSIGNMENT_SIZE_MISMATCH,
    SLICE_ASSIGNMENT_NOT_2D,
    SLICE_ASSIGNMENT_ROWS_MISMATCH,
    SLICE_ASSIGNMENT_COLS_MISMATCH,
    CORE_INDEX_OUT_OF_RANGE
)
from netochi.mapping.exceptions import MappingValidationError

# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------
MAPPING_STATE = TypeVar('MAPPING_STATE', bound='MappingState')
ANY_MAPPING_INPUT = TypeVar('ANY_MAPPING_INPUT', bound=MappingInput)
WITH_HW_INPUT = TypeVar('WITH_HW_INPUT', bound=MappingInput)

# -----------------------------------------------------------------------------
# Mapping State Interfaces
# -----------------------------------------------------------------------------

class MappingState(BaseModel, Generic[ANY_MAPPING_INPUT]):
    """Base class for all mapping results, encapsulating the original input."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    mapping_input: ANY_MAPPING_INPUT


class MosaicMappingState(MappingState[MosaicMappingInput[Any]]):
    """Specific mapping result for Mosaic hardware with detailed assignments."""
    
    # 1D array: Mapping of [neuron_idx] -> core_idx
    neuron_core_idxs_assignment: np.ndarray 
    
    # 1D array: Mapping of [neuron_idx] -> local_neuron_idx (within core)
    neuron_local_idxs_assignment: np.ndarray

    # 2D array: Mapping of [neuron_idx, router_level] -> slice_idx
    neuron_slice_assignments: np.ndarray

    @model_validator(mode='after')
    def validate_mosaic_assignments(self) -> 'MosaicMappingState':
        """Ensure that the assignments are consistent with the hardware and graph."""
        hw = self.mapping_input.hw_config
        num_neurons = self.mapping_input.graph.num_vertices()
        
        # 1. Shape Verification
        if self.neuron_core_idxs_assignment.ndim != 1:
            raise MappingValidationError(CORE_ASSIGNMENT_NOT_1D)
        
        if self.neuron_core_idxs_assignment.size != num_neurons:
            raise MappingValidationError(
                CORE_ASSIGNMENT_SIZE_MISMATCH.format(
                    actual=self.neuron_core_idxs_assignment.size, 
                    expected=num_neurons
                )
            )

        if self.neuron_local_idxs_assignment.ndim != 1:
            raise MappingValidationError("neuron_local_idxs_assignment must be a 1D array")
        
        if self.neuron_local_idxs_assignment.size != num_neurons:
            raise MappingValidationError("Local index assignment size must match number of neurons")
            
        if self.neuron_slice_assignments.ndim != 2:
            raise MappingValidationError(SLICE_ASSIGNMENT_NOT_2D)
            
        if self.neuron_slice_assignments.shape[0] != num_neurons:
            raise MappingValidationError(SLICE_ASSIGNMENT_ROWS_MISMATCH)
            
        if self.neuron_slice_assignments.shape[1] != hw.router_levels + 1:
            raise MappingValidationError(
                SLICE_ASSIGNMENT_COLS_MISMATCH.format(
                    actual=self.neuron_slice_assignments.shape[1], 
                    expected=hw.router_levels + 1
                )
            )

        # 2. Value Range Verification
        if np.any(self.neuron_core_idxs_assignment < 0) or np.any(self.neuron_core_idxs_assignment >= hw.total_cores):
            raise MappingValidationError(
                CORE_INDEX_OUT_OF_RANGE.format(max_cores=hw.total_cores)
            )

        if np.any(self.neuron_local_idxs_assignment < 0) or np.any(self.neuron_local_idxs_assignment >= hw.neurons_per_core):
            raise MappingValidationError("One or more local indices are out of valid range")
        
        return self

# -----------------------------------------------------------------------------
# Mapper Interface
# -----------------------------------------------------------------------------

class BaseMapper(ABC, Generic[MAPPING_STATE, WITH_HW_INPUT]):
    """Structural interface for mapping algorithms."""
    
    def get_name(self) -> str:
        """Return the name of the mapping algorithm."""
        return self.__class__.__name__

    @abstractmethod
    def run(self, mapping_input: WITH_HW_INPUT) -> MAPPING_STATE:
        """
        Execute the mapping algorithm on the provided input.
        Returns a MappingState object containing the results.
        """
        pass
