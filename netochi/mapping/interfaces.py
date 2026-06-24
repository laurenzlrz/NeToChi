from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Any

import numpy as np
import numpy.typing as npt
import icontract

from netochi.definitions.freezable import freezable, Freezable
from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, HWMappingInput, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# -----------------------------------------------------------------------------
# Base Mapping State Interfaces
# -----------------------------------------------------------------------------

@freezable
class MappingState[ANY_MAPPING_INPUT: MappingInput, HW_CONFIG: Any](Freezable):
    """Base class for all mapping results."""
    
    def __init__(
        self,
        mapping_input: Optional[ANY_MAPPING_INPUT] = None,
    ) -> None:
        self.mapping_input = mapping_input
        self.freeze()

    @property
    @abstractmethod
    def hw_to_evaluate(self) -> HW_CONFIG:
        pass


@freezable
class HWNetworkMappingState[ANY_MAPPING_INPUT: MappingInput, INFERRED_HW_CONFIG: Any](MappingState[ANY_MAPPING_INPUT, INFERRED_HW_CONFIG]):
    """
    Base class for states that infer hardware. 
    Does not strictly require hardware parameters in input, but provides/infers them.
    """
    
    def __init__(
        self,
        mapping_input: Optional[ANY_MAPPING_INPUT] = None,
        inferred_hw: Optional[INFERRED_HW_CONFIG] = None,
    ) -> None:
        super().__init__(mapping_input=mapping_input)
        self.unfreeze()
        self.inferred_hw = inferred_hw
        self.freeze()

    @property
    def hw_to_evaluate(self) -> INFERRED_HW_CONFIG:
        """For HW-aware mappers, the hardware to evaluate is the inferred hardware."""
        return self.inferred_hw


@freezable
class NetworkAssignmentState[WITH_HW_INPUT: HWMappingInput, GT_HW_CONFIG: Any](MappingState[WITH_HW_INPUT, GT_HW_CONFIG]):
    """
    State for hardware-aware partitioning.
    Requires specific hardware parameters in the input (WITH_HW_INPUT).
    """
    
    def __init__(
        self,
        mapping_input: Optional[WITH_HW_INPUT] = None,
    ) -> None:
        super().__init__(mapping_input=mapping_input)

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

@freezable
class BaseMosaicMappingState[ANY_MAPPING_INPUT: MappingInput](MappingState[ANY_MAPPING_INPUT, MosaicHardwareConfig]):
    """
    Abstract base state for all Mosaic mappers. Infers HW.
    Contains the assignment arrays and uses HWNetworkMappingState to ensure mapping_input exists.
    """
    
    def __init__(
        self,
        assignment: MosaicAssignment,
        mapping_input: Optional[ANY_MAPPING_INPUT] = None,
    ) -> None:
        MappingState.__init__(self, mapping_input=mapping_input)
        self.unfreeze()
        self.assignment = assignment
        self.freeze()

    @property
    def c(self) -> npt.NDArray[np.int_]:
        return self.assignment.neuron_core_pre_assignment

    @property
    def x(self) -> npt.NDArray[np.int_]:
        return self.assignment.neuron_idx_pre_assignment
    
    @property
    def s(self) -> np.ndarray[tuple[Any, Any], np.dtype[np.int_]]:
        return self.assignment.neuron_slice_assignment


@freezable
@icontract.invariant(lambda self: self.validate())
class MosaicNetworkMappingState(BaseMosaicMappingState[MosaicMappingInput], NetworkAssignmentState[MosaicMappingInput, MosaicHardwareConfig]):
    """
    State for pure partitioning/assignment mappers. 
    Input MUST contain the hardware configuration (MosaicMappingInput).
    """

    def __init__(
        self,
        assignment: MosaicAssignment,
        mapping_input: Optional[MosaicMappingInput] = None,
    ) -> None:
        super().__init__(assignment=assignment, mapping_input=mapping_input)

    def validate(self) -> bool:
        self.mapping_input.hw_config.verify_assignment(self.assignment)
        return True

    @classmethod
    def from_input_zero(cls, mapping_input: MosaicMappingInput) -> 'MosaicNetworkMappingState':
        hw = mapping_input.hw_config
        return cls(
            mapping_input=mapping_input,
            assignment=MosaicAssignment.spread(num_neurons=mapping_input.graph.num_vertices(), hw=hw)
        )

    @classmethod
    def from_input_random(cls, mapping_input: MosaicMappingInput, seed: Optional[int] = None) -> 'MosaicNetworkMappingState':
        hw = mapping_input.hw_config
        return cls(
            mapping_input=mapping_input,
            assignment=MosaicAssignment.random(num_neurons=mapping_input.graph.num_vertices(), hw=hw, seed=seed)
        )


@freezable
@icontract.invariant(lambda self: self.validate())
class MosaicHWMappingState[ANY_MAPPING_INPUT: MappingInput](BaseMosaicMappingState[ANY_MAPPING_INPUT], HWNetworkMappingState[ANY_MAPPING_INPUT, MosaicHardwareConfig]):
    """
    State for joint inference mappers. 
    Input is purely the network (MappingInput), but output includes the optimized hardware.
    """

    def __init__(
        self,
        assignment: MosaicAssignment,
        mapping_input: Optional[ANY_MAPPING_INPUT] = None,
        inferred_hw: Optional[MosaicHardwareConfig] = None,
    ) -> None:
        BaseMosaicMappingState.__init__(self, assignment=assignment, mapping_input=mapping_input)
        HWNetworkMappingState.__init__(self, mapping_input=mapping_input, inferred_hw=inferred_hw)

    def validate(self) -> bool:
        self.inferred_hw.verify_assignment(self.assignment)
        return True

    @classmethod
    def from_guess_zero(cls, mapping_input: ANY_MAPPING_INPUT, initial_hw_guess: MosaicHardwareConfig) -> 'MosaicHWMappingState[ANY_MAPPING_INPUT]':
        """
        Creates an uninitialized state using an initial hardware guess to dimension the arrays.
        This guess is generated by the mapper and does NOT leak ground truth information.
        """
        return cls(
            mapping_input=mapping_input,
            inferred_hw=initial_hw_guess,
            assignment=MosaicAssignment.spread(
                hw=initial_hw_guess,
                num_neurons=mapping_input.graph.num_vertices()
            )
        )

    @classmethod
    def from_guess_random(cls, mapping_input: ANY_MAPPING_INPUT, initial_hw_guess: MosaicHardwareConfig, seed: Optional[int]) -> 'MosaicHWMappingState[ANY_MAPPING_INPUT]':
        return cls(
            mapping_input=mapping_input,
            inferred_hw=initial_hw_guess,
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
