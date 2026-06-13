from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING, Optional, Any

from pydantic import BaseModel, ConfigDict

from netochi.mapping.interfaces import BaseMosaicMappingState

if TYPE_CHECKING:
    from netochi.pipeline.results import PipelineSummary

class MappingMetric[MAPPING_STATE, MAPPING_STATE_BASELINE](ABC, BaseModel):
    """
    Abstract interface for mapping metrics.
    Metrics evaluate a final mapping state, potentially against a baseline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

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
    def run(self) -> 'PipelineSummary':
        """Execute the pipeline and return a summary of results."""
        pass
