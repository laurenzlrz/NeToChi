from typing import TypeVar, Generic, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict, model_validator
from netochi.input_generator.interfaces import MappingInput, MosaicHWMappingInput, WITH_HW_INPUT
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig

# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------
ANY_MAPPING_INPUT = TypeVar('ANY_MAPPING_INPUT', bound=MappingInput[Any])
PAYLOAD = TypeVar('PAYLOAD')

# -----------------------------------------------------------------------------
# Base Mapping State Interfaces
# -----------------------------------------------------------------------------

class MappingState(BaseModel, Generic[ANY_MAPPING_INPUT]):
    """Base class for all mapping results."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False) # only for pydantic

    def init_random_assignments(self, seed: Optional[int] = None) -> None:
        """Abstract initialization method for assignments.
        baseically constructor"""
        pass

class HWNetworkMappingState(MappingState[ANY_MAPPING_INPUT], Generic[ANY_MAPPING_INPUT]):
    """
    Base class for states that infer hardware. 
    Does not strictly require hardware parameters in input, but provides/infers them.
    """
    mapping_input: ANY_MAPPING_INPUT

    def _init_random_hw(self, seed: Optional[int] = None) -> None:
        """Randomly initialize hardware configuration."""
        pass

class NetworkAssignmentState(MappingState[WITH_HW_INPUT], Generic[WITH_HW_INPUT]):
    """
    State for hardware-aware partitioning. 
    Requires specific hardware parameters in the input (WITH_HW_INPUT).
    """
    mapping_input: WITH_HW_INPUT

# -----------------------------------------------------------------------------
# Mosaic Specific Interfaces
# -----------------------------------------------------------------------------

class BaseMosaicMappingState(MappingState[ANY_MAPPING_INPUT], Generic[ANY_MAPPING_INPUT]):
    """
    Abstract base state for all Mosaic mappers.  Infers HW
    Contains the assignment arrays and uses HWNetworkMappingState to ensure mapping_input exists.
    """
    neuron_core_idxs_assignment: npt.NDArray[np.int_] = Field(description="[neuron_idx] -> core_idx")
    neuron_local_idxs_assignment: npt.NDArray[np.int_] = Field(description="[neuron_idx] -> local_neuron_idx")
    neuron_slice_assignments: npt.NDArray[np.int_] = Field(description="[neuron_idx, dist] -> slice_idx")

    @property
    def c(self) -> npt.NDArray[np.int_]: return self.neuron_core_idxs_assignment
    
    @property
    def x(self) -> npt.NDArray[np.int_]: return self.neuron_local_idxs_assignment
    
    @property
    def s(self) -> npt.NDArray[np.int_]: return self.neuron_slice_assignments

    @model_validator(mode='after')
    def validate_mosaic_assignments(self) -> 'BaseMosaicMappingState[ANY_MAPPING_INPUT]':
        try:
            hw = self.hw
        except (NotImplementedError, AttributeError):
            return self

        num_neurons: int = self.mapping_input.graph.num_vertices()
        if self.neuron_core_idxs_assignment.size != num_neurons:
            raise ValueError(f"Core assignment size mismatch: {self.neuron_core_idxs_assignment.size} != {num_neurons}")
        if self.neuron_core_idxs_assignment.ndim != 1:
            raise ValueError("Core assignment must be 1D")
        return self

class MosaicNetworkMappingState(BaseMosaicMappingState[MosaicHWMappingInput[PAYLOAD]], NetworkAssignmentState[MosaicHWMappingInput[PAYLOAD]], Generic[PAYLOAD]):
    """
    State for pure partitioning/assignment mappers. 
    Input MUST contain the hardware configuration (MosaicMappingInput).
    """

    @classmethod
    def from_input(cls, mapping_input: MosaicHWMappingInput[PAYLOAD]) -> 'MosaicNetworkMappingState[PAYLOAD]':
        hw = mapping_input.hw_config
        N: int = mapping_input.graph.num_vertices()
        return cls(
            mapping_input=mapping_input,
            neuron_core_idxs_assignment=np.zeros(N, dtype=int),
            neuron_local_idxs_assignment=np.zeros(N, dtype=int),
            neuron_slice_assignments=np.zeros((N, hw.max_distance + 1), dtype=int)
        )

    def init_random_assignments(self, seed: Optional[int] = None) -> None:
        """Randomly initialize assignments respecting given hardware capacities."""
        hw = self.hw
        N: int = self.mapping_input.graph.num_vertices()
        rng = np.random.default_rng(seed)
        
        slots = [(c, x) for c in range(hw.total_cores) for x in range(hw.neurons_per_core)]
        rng.shuffle(slots)
        
        for i in range(N):
            self.neuron_core_idxs_assignment[i], self.neuron_local_idxs_assignment[i] = slots[i]
            
        for d in range(1, hw.max_distance + 1):
            n_sl: int = hw.num_slices_at_distance(d)
            self.neuron_slice_assignments[:, d] = rng.integers(0, n_sl, size=N)

class MosaicHWMappingState(BaseMosaicMappingState[ANY_MAPPING_INPUT], HWNetworkMappingState[ANY_MAPPING_INPUT], Generic[ANY_MAPPING_INPUT, PAYLOAD]):
    """
    State for joint inference mappers. 
    Input is purely the network (MappingInput), but output includes the optimized hardware.
    """
    hw_config: MosaicHardwareConfig

    @property
    def hw(self) -> MosaicHardwareConfig:
        return self.hw_config

    @classmethod
    def create_uninitialized_state(cls, mapping_input: ANY_MAPPING_INPUT, initial_hw_guess: MosaicHardwareConfig) -> 'MosaicHWMappingState[ANY_MAPPING_INPUT, PAYLOAD]':
        """
        Creates an uninitialized state using an initial hardware guess to dimension the arrays.
        This guess is generated by the mapper and does NOT leak ground truth information.
        """
        N: int = mapping_input.graph.num_vertices()
        return cls(
            mapping_input=mapping_input,
            hw_config=initial_hw_guess,
            neuron_core_idxs_assignment=np.zeros(N, dtype=int),
            neuron_local_idxs_assignment=np.zeros(N, dtype=int),
            neuron_slice_assignments=np.zeros((N, initial_hw_guess.max_distance + 1), dtype=int)
        )

    def init_random_assignments(self, seed: Optional[int] = None) -> None:
        """Randomly initialize assignments respecting the learned hardware capacities."""
        hw = self.hw
        N: int = self.mapping_input.graph.num_vertices()
        rng = np.random.default_rng(seed)
        
        slots = [(c, x) for c in range(hw.total_cores) for x in range(hw.neurons_per_core)]
        rng.shuffle(slots)
        
        for i in range(N):
            self.neuron_core_idxs_assignment[i], self.neuron_local_idxs_assignment[i] = slots[i]
            
        for d in range(1, hw.max_distance + 1):
            n_sl: int = hw.num_slices_at_distance(d)
            self.neuron_slice_assignments[:, d] = rng.integers(0, n_sl, size=N)

# -----------------------------------------------------------------------------
# Mapper Interface
# -----------------------------------------------------------------------------

MAPPING_STATE = TypeVar('MAPPING_STATE', bound=MappingState[Any])

class BaseMapper(ABC, Generic[MAPPING_STATE, ANY_MAPPING_INPUT]):
    """Structural interface for mapping algorithms."""
    
    def get_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(self, mapping_input: ANY_MAPPING_INPUT) -> MAPPING_STATE:
        pass
