from typing import Dict, Any, TypeVar, Generic, Optional
import graph_tool.all as gt
import numpy as np
from pydantic.dataclasses import dataclass
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig

# -----------------------------------------------------------------------------
# Type Variables for Generics
# -----------------------------------------------------------------------------
PAYLOAD = TypeVar("PAYLOAD")
WITH_HW_INPUT = TypeVar("WITH_HW_INPUT", bound=MappingInput)

# -----------------------------------------------------------------------------
# Base Interfaces
# -----------------------------------------------------------------------------

class BaseInputFactory:
    """Abstract base class for factories that generate MappingInputs."""
    def generate(self) -> MappingInput:
        """
        Returns a MappingInput object.
        """
        raise NotImplementedError

@dataclass
class MappingInput(Generic[PAYLOAD]):
    """Base model for all experiment inputs, generic over payload."""
    graph: gt.Graph
    descriptions: Dict[str, str]
    payload: Optional[PAYLOAD] = None


@dataclass
class MosaicMappingInput(MappingInput[PAYLOAD], Generic[PAYLOAD]):
    """Problem input with a predefined hardware configuration."""
    hw_config: MosaicHardwareConfig
    pre_assignment: Optional[np.ndarray] = None