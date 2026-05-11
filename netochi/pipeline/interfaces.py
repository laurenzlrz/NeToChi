from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict
from netochi.mapping.interfaces import MappingState

if TYPE_CHECKING:
    from netochi.pipeline.results import PipelineSummary

MAPPING_STATE = TypeVar("MAPPING_STATE", bound=MappingState)

class MappingMetric(BaseModel, ABC, Generic[MAPPING_STATE]):
    """
    Abstract interface for mapping metrics.
    Metrics evaluate a final mapping state and return a scalar value.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @abstractmethod
    def evaluate(self, state: MAPPING_STATE) -> float:
        """Evaluate the state and return a score."""
        pass

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

class BasePipelineRunner(ABC):
    """
    Structural interface for the benchmarking pipeline.
    """
    @abstractmethod
    def run(self) -> 'PipelineSummary':
        """Execute the pipeline and return a summary of results."""
        pass
