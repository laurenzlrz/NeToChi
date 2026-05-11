from typing import TypeVar, Generic, Optional
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator
from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput

# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------
MAPPING_STATE = TypeVar('MAPPING_STATE', bound='MappingState')
ANY_MAPPING_INPUT = TypeVar('ANY_MAPPING_INPUT', bound=MappingInput)
WITH_HW_INPUT = TypeVar('WITH_HW_INPUT', bound=MappingInput)
PAYLOAD = TypeVar('PAYLOAD')

# -----------------------------------------------------------------------------
# Mapping State Interfaces
# -----------------------------------------------------------------------------

class MappingState(BaseModel, Generic[ANY_MAPPING_INPUT]):
    """Base class for all mapping results, encapsulating the original input."""
    # Set frozen=False to allow inplace modifications during optimization
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
    mapping_input: ANY_MAPPING_INPUT

    def init_random(self, seed: Optional[int] = None):
        """Abstract initialization method."""
        pass


class MosaicMappingState(MappingState[MosaicMappingInput[PAYLOAD]], Generic[PAYLOAD]):
    """
    Specific mapping result for Mosaic hardware with detailed assignments.
    
    This is the UNIFIED state object used for both immutable results 
    and mutable optimization (MCMC).
    """
    
    # Assignments as NumPy arrays
    neuron_core_idxs_assignment: np.ndarray = Field(description="[neuron_idx] -> core_idx")
    neuron_local_idxs_assignment: np.ndarray = Field(description="[neuron_idx] -> local_neuron_idx")
    neuron_slice_assignments: np.ndarray = Field(description="[neuron_idx, dist] -> slice_idx")

    # --- Convenience Aliases for performance loops ---
    @property
    def c(self) -> np.ndarray: return self.neuron_core_idxs_assignment
    
    @property
    def x(self) -> np.ndarray: return self.neuron_local_idxs_assignment
    
    @property
    def s(self) -> np.ndarray: return self.neuron_slice_assignments

    @model_validator(mode='after')
    def validate_mosaic_assignments(self) -> 'MosaicMappingState':
        """Ensure that the assignments are consistent with the hardware and graph."""
        hw = self.mapping_input.hw_config
        num_neurons = self.mapping_input.graph.num_vertices()
        
        if self.neuron_core_idxs_assignment.size != num_neurons:
            raise ValueError(f"Core assignment size mismatch: {self.neuron_core_idxs_assignment.size} != {num_neurons}")
        
        if self.neuron_core_idxs_assignment.ndim != 1:
            raise ValueError("Core assignment must be 1D")
            
        return self

    @classmethod
    def from_input(cls, mapping_input: MosaicMappingInput[PAYLOAD]) -> 'MosaicMappingState[PAYLOAD]':
        """Initialize a new state with zeroed assignments."""
        hw = mapping_input.hw_config
        N = mapping_input.graph.num_vertices()
        return cls(
            mapping_input=mapping_input,
            neuron_core_idxs_assignment=np.zeros(N, dtype=int),
            neuron_local_idxs_assignment=np.zeros(N, dtype=int),
            neuron_slice_assignments=np.zeros((N, hw.max_distance + 1), dtype=int)
        )

    def init_random(self, seed: Optional[int] = None):
        """Randomly initialize assignments respecting hardware capacities."""
        hw = self.mapping_input.hw_config
        N = self.mapping_input.graph.num_vertices()
        rng = np.random.default_rng(seed)
        
        # 1. Distribute across cores evenly
        slots = [(c, x) for c in range(hw.total_cores) for x in range(hw.neurons_per_core)]
        rng.shuffle(slots)
        
        for i in range(N):
            self.neuron_core_idxs_assignment[i], self.neuron_local_idxs_assignment[i] = slots[i]
            
        # 2. Randomize slices
        for d in range(1, hw.max_distance + 1):
            n_sl = hw.num_slices_at_distance(d)
            self.neuron_slice_assignments[:, d] = rng.integers(0, n_sl, size=N)

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
