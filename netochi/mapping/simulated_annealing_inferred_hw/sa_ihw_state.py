from typing import Optional, TYPE_CHECKING, Any
import numpy as np
import numpy.typing as npt

from netochi.input_generator.interfaces import MappingInput, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.simulated_annealing_fix_hw.sa_state import SAState
from netochi.mapping.interfaces import BaseMosaicMappingState

if TYPE_CHECKING:
    from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner


class SAIHWState(SAState, BaseMosaicMappingState[MappingInput]):
    """
    State for Simulated Annealing with Inferred Hardware.
    Encapsulates a MosaicAssignment and the mapping input.
    Subclasses SAState and BaseMosaicMappingState for polymorphism.
    """

    def __init__(
        self,
        mapping_input: MappingInput,
        initial_hw: MosaicHardwareConfig,
        assignment: MosaicAssignment
    ):
        # Initialize BaseMosaicMappingState
        BaseMosaicMappingState.__init__(
            self,
            _mapping_input=mapping_input,
            assignment=assignment
        )

        self.hw_config = initial_hw
        self.core_assignment = np.asarray(assignment.neuron_core_pre_assignment, dtype=np.int_)
        self.local_assignment = np.asarray(assignment.neuron_idx_pre_assignment, dtype=np.int_)
        slot_to_node = np.full((initial_hw.total_cores, initial_hw.neurons_per_core), -1, dtype=np.int_)
        slot_to_node[self.core_assignment, self.local_assignment] = np.arange(len(self.core_assignment))
        self.slot_to_node = slot_to_node
        self._slice_assigner_val: Optional['DeltaOptimalSliceAssigner'] = None

    @property
    def _slice_assigner(self) -> Optional['DeltaOptimalSliceAssigner']:
        return self._slice_assigner_val

    @_slice_assigner.setter
    def _slice_assigner(self, value: Optional['DeltaOptimalSliceAssigner']) -> None:
        self._slice_assigner_val = value

    @property
    def K(self) -> int:
        return self.hw_config.total_cores

    @property
    def Nc(self) -> int:
        return self.hw_config.neurons_per_core

    @property
    def Nr(self) -> int:
        return self.hw_config.nodes_per_router

    @property
    def L(self) -> int:
        return self.hw_config.router_levels

    @property
    def slice_factor(self) -> int:
        return self.hw_config.slice_factor

    @property
    def hw_ground_truth(self) -> MosaicHardwareConfig:
        return self.hw_config

    @property
    def hw_inferred(self) -> MosaicHardwareConfig:
        return self.hw_config

    @property
    def hw_to_evaluate(self) -> MosaicHardwareConfig:
        return self.hw_config

    @property
    def c(self) -> npt.NDArray[np.int_]:
        return self.core_assignment

    @property
    def x(self) -> npt.NDArray[np.int_]:
        return self.local_assignment

    @property
    def s(self) -> npt.NDArray[np.int_]:
        assert self._slice_assigner is not None
        return self._slice_assigner.slice_assignment

    def update_assignment(self, new_assignment: MosaicAssignment) -> None:
        """Updates the core/local/slice assignment."""
        self.assignment = new_assignment
        self.hw_config = new_assignment.hw
        self.core_assignment = np.asarray(new_assignment.neuron_core_pre_assignment, dtype=np.int_)
        self.local_assignment = np.asarray(new_assignment.neuron_idx_pre_assignment, dtype=np.int_)

        slot_to_node = np.full((self.K, self.Nc), -1, dtype=np.int_)
        slot_to_node[self.core_assignment, self.local_assignment] = np.arange(len(self.core_assignment))
        self.slot_to_node = slot_to_node
