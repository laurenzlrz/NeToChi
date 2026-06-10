from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING, Optional, Any

from netochi.mapping.interfaces import BaseMosaicMappingState

if TYPE_CHECKING:
    from netochi.pipeline.results import PipelineSummary

MAPPING_STATE_CONTRA = TypeVar("MAPPING_STATE_CONTRA", bound=BaseMosaicMappingState[Any], contravariant=True)
MAPPING_STATE2_CONTRA = TypeVar("MAPPING_STATE2_CONTRA", bound=BaseMosaicMappingState[Any], contravariant=True)

class MappingMetric(ABC, Generic[MAPPING_STATE_CONTRA, MAPPING_STATE2_CONTRA]):
    """
    Abstract interface for mapping metrics.
    Metrics evaluate a final mapping state, potentially against a baseline.
    """

    @abstractmethod
    def evaluate_against_baseline(self, state: MAPPING_STATE_CONTRA, baseline: Optional[MAPPING_STATE2_CONTRA] = None) -> float:
        """
        Evaluate the state and return a score.
        If a baseline is provided, the metric may perform comparative evaluation.
        """
        pass

    @abstractmethod
    def evaluate(self, state: MAPPING_STATE_CONTRA) -> float:
        """
        Evaluate the state and return an absolute score.
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
