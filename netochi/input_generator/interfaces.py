from typing import Dict, Any, TypeVar, Generic, Optional
import graph_tool.all as gt
import numpy as np
import numpy.typing as npt
from pydantic.dataclasses import dataclass
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig

# -----------------------------------------------------------------------------
# Base Models
# -----------------------------------------------------------------------------

PAYLOAD = TypeVar("PAYLOAD")
HW_CONFIG = TypeVar("HW_CONFIG")

@dataclass(config=dict(arbitrary_types_allowed=True), kw_only=True)
class MappingInput(Generic[PAYLOAD]):
    """Base model for all experiment inputs, generic over payload."""
    graph: gt.Graph
    descriptions: Dict[str, str]
    payload: Optional[PAYLOAD] = None

@dataclass(config=dict(arbitrary_types_allowed=True), kw_only=True)
class HWMappingInput(MappingInput[PAYLOAD], Generic[PAYLOAD, HW_CONFIG]):
    """Base model for all experiment inputs, generic over payload."""
    hw_config: HW_CONFIG

@dataclass(config=dict(arbitrary_types_allowed=True), kw_only=True)
class MosaicMappingInput(HWMappingInput[PAYLOAD, MosaicHardwareConfig], Generic[PAYLOAD]):
    """Problem input with a predefined hardware configuration."""
    pre_assignment: Optional[npt.NDArray[np.int_]] = None

# -----------------------------------------------------------------------------
# Type Variables for Factory
# -----------------------------------------------------------------------------

MAPPING_INPUT = TypeVar("MAPPING_INPUT", bound=HWMappingInput[Any, Any])
WITH_HW_INPUT = TypeVar("WITH_HW_INPUT", bound=HWMappingInput[Any, Any])

# -----------------------------------------------------------------------------
# Base Interfaces
# -----------------------------------------------------------------------------

class BaseInputFactory(Generic[MAPPING_INPUT]):
    """Abstract base class for factories that generate MappingInputs."""
    def generate(self) -> MAPPING_INPUT:
        """
        Returns a MappingInput object.
        """
        raise NotImplementedError

class HWBaseInputFactory(BaseInputFactory[WITH_HW_INPUT], Generic[WITH_HW_INPUT]):
    """Abstract base class for factories that generate MappingInputs."""
    def generate(self) -> WITH_HW_INPUT:
        """
        Returns a MappingInput object.
        """
        raise NotImplementedError