from abc import ABC, abstractmethod
from typing import Generic, Optional

from pydantic import BaseModel, ConfigDict

from netochi.definitions.generics import Input_contra, MappingState_contra, BaselineState_contra, Input_co, \
    MappingState_co, BaselineState_co
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


class BasePipelineRunner(ABC, BaseModel, Generic[Input_co, MappingState_co, BaselineState_co]):
    """
    Structural interface for the benchmarking pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def run(self) -> list[PipelineSummary[Input_co, MappingState_co, BaselineState_co]]:
        """Execute the pipeline and return a summary of results."""
        pass

# TODO Put generics into another folder

class PipelineConsumer(ABC, BaseModel, Generic[Input_contra, MappingState_contra, BaselineState_contra]):

    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    def consume(self, data: PipelineSummary[Input_contra, MappingState_contra, BaselineState_contra]) -> None:
        NotImplementedError("Consume method must be implemented by subclasses of PipelineConsumer.")