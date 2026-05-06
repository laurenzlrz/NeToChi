from typing import Iterator, Tuple, Dict, Any, Protocol, runtime_checkable
import graph_tool.all as gt
from pydantic import BaseModel, Field, ConfigDict
from netochi.mapping.hardware_config import HardwareConfig
from netochi.mapping.likelihood_state import MappingState, MappingResult
from netochi.pipeline.objectives import MappingObjective

# -----------------------------------------------------------------------------
# Core Protocols (Interfaces)
# -----------------------------------------------------------------------------

@runtime_checkable
class Metric(Protocol):
    """Structural interface for evaluating mapping quality."""
    def get_name(self) -> str: ...
    def evaluate(self, state: MappingState) -> float: ...

@runtime_checkable
class PipelineStage(Protocol):
    """A discrete stage in the benchmarking pipeline."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]: ...


