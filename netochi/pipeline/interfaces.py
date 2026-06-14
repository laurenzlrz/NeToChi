from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING, Optional, Any

from pydantic import BaseModel, ConfigDict

from netochi.pipeline import PipelineSummary

if TYPE_CHECKING:
    from netochi.pipeline.results import PipelineSummary

class MappingMetric[MAPPING_STATE, MAPPING_STATE_BASELINE](ABC):
    """
    Abstract interface for mapping metrics.
    Metrics evaluate a final mapping state, potentially against a baseline.
    """

    @abstractmethod
    def evaluate_against_baseline(self, state: MAPPING_STATE, baseline: Optional[MAPPING_STATE_BASELINE] = None) -> float:
        """
        Evaluate the state and return a score.
        If a baseline is provided, the metric may perform comparative evaluation.
        """
        pass

    @abstractmethod
    def evaluate(self, state: MAPPING_STATE) -> float:
        """
        Evaluate the state and return an absolute score.
        """
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class BasePipelineRunner(ABC, BaseModel):
    """
    Structural interface for the benchmarking pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def run(self) -> list['PipelineSummary']:
        """Execute the pipeline and return a summary of results."""
        pass


class PipelineConsumer(ABC, BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    @abstractmethod
    def consume(self, data: PipelineSummary) -> None:
        """
        Consume the pipeline summary data.
        This method should be implemented by subclasses to define specific consumption behavior.
        """
        pass
