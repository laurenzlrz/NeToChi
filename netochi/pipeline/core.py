from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Dict, Any, TypeVar
import graph_tool.all as gt
from netochi.mapping.hardware_config import HardwareConfig
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.objectives import MappingObjective

# -----------------------------------------------------------------------------
# Mapper Capability Interfaces
# -----------------------------------------------------------------------------

class BaseMapper(ABC):
    """Base class for all mapping algorithms."""
    @classmethod
    def get_name(cls) -> str:
        """Return the class name for identification."""
        return cls.__name__

class IFixedHardwareMapper(ABC):
    """Interface for mappers targeting a fixed hardware configuration."""
    @abstractmethod
    def map_fixed_hardware(self, mapping_input: 'FixedHardwareInput') -> MappingState:
        """Execute mapping onto fixed hardware."""
        pass

class IVariableHardwareMapper(ABC):
    """Interface for mappers that can optimize hardware parameters."""
    @abstractmethod
    def map_variable_hardware(self, mapping_input: 'VariableHardwareInput') -> MappingState:
        """Execute mapping with hardware flexibility."""
        pass

# -----------------------------------------------------------------------------
# Input Hierarchy (Visitor Pattern)
# -----------------------------------------------------------------------------

class MappingInput(ABC):
    """Base class for all experiment inputs."""
    def __init__(self, graph: gt.Graph, objective: MappingObjective = None, metadata: Dict[str, Any] = None):
        """Initialize with network graph and optional objective/metadata."""
        self.graph = graph
        self.objective = objective
        self.metadata = metadata or {}

    @abstractmethod
    def accept(self, mapper: BaseMapper) -> MappingState:
        """Visitor dispatch point for mapper capability check."""
        pass

class FixedHardwareInput(MappingInput):
    """Problem input with a predefined hardware configuration."""
    def __init__(self, graph: gt.Graph, hw_config: HardwareConfig, objective: MappingObjective = None, metadata: Dict[str, Any] = None):
        """Initialize with graph, fixed hardware config, and optional objective."""
        super().__init__(graph, objective, metadata)
        self.hw_config = hw_config

    def accept(self, mapper: BaseMapper) -> MappingState:
        """Dispatch to map_fixed_hardware if capability exists."""
        if isinstance(mapper, IFixedHardwareMapper):
            return mapper.map_fixed_hardware(self)
        raise NotImplementedError(f"Mapper '{mapper.get_name()}' does not support FixedHardwareInput (requires IFixedHardwareMapper).")

class VariableHardwareInput(MappingInput):
    """Problem input where hardware optimization is allowed."""
    def accept(self, mapper: BaseMapper) -> MappingState:
        """Dispatch to map_variable_hardware if capability exists."""
        if isinstance(mapper, IVariableHardwareMapper):
            return mapper.map_variable_hardware(self)
        raise NotImplementedError(f"Mapper '{mapper.get_name()}' does not support VariableHardwareInput (requires IVariableHardwareMapper).")


# -----------------------------------------------------------------------------
# Factory and Metric Definitions
# -----------------------------------------------------------------------------

class BaseInputFactory(ABC):
    """Abstract base class for factories that generate MappingInputs."""
    @abstractmethod
    def generate(self) -> Iterator[Tuple[MappingInput, Dict[str, Any]]]:
        """
        Yields a tuple containing the MappingInput and a dictionary of metadata.
        """
        pass

class BaseMetric(ABC):
    """Abstract base class for evaluating mapping quality."""
    @abstractmethod
    def evaluate(self, state: MappingState) -> float:
        """Evaluates the mapping state and returns a scalar metric."""
        pass
        
    @classmethod
    def get_name(cls) -> str:
        """Return the name of the metric for reporting."""
        return cls.__name__
