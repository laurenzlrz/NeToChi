from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING, Optional, Any
from pydantic import BaseModel, ConfigDict
from netochi.mapping.interfaces import MappingState

if TYPE_CHECKING:
    from netochi.pipeline.results import PipelineSummary

MAPPING_STATE = TypeVar("MAPPING_STATE", bound=MappingState[Any])
MAPPING_STATE2 = TypeVar("MAPPING_STATE2", bound=MappingState[Any])

class MappingMetric(ABC, Generic[MAPPING_STATE, MAPPING_STATE2]):
    """
    Abstract interface for mapping metrics.
    Metrics evaluate a final mapping state, potentially against a baseline.
    """

    @abstractmethod
    def evaluate(self, state: MAPPING_STATE, baseline: Optional[MAPPING_STATE2] = None) -> float:
        """
        Evaluate the state and return a score.
        If a baseline is provided, the metric may perform comparative evaluation.
        """
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

class BasePipelineRunner(ABC):
    """
    Structural interface for the benchmarking pipeline.
    """
    @abstractmethod
    def run(self) -> 'PipelineSummary':
        """Execute the pipeline and return a summary of results."""
        pass
