from abc import ABC, abstractmethod
from typing import Generic, Optional

from pydantic import BaseModel, ConfigDict, Field

from netochi.pipeline.config import PipelineOutput
from netochi.mapping.interfaces import MappingState
from netochi.definitions.generics import Input_contra, MappingState_contra, BaselineState_contra, Input_co, \
    MappingState_co, BaselineState_co
from netochi.pipeline.results import PipelineSummary
from netochi.objectives.interfaces import ObjectiveResult

class MappingMetric[MAPPING_STATE, MAPPING_STATE_BASELINE](ABC):
    """
    Abstract interface for mapping metrics.
    Metrics evaluate a final mapping state, potentially against a baseline.
    """

    @abstractmethod
    def evaluate(self, state: MAPPING_STATE, baseline: Optional[MAPPING_STATE_BASELINE] = None) -> ObjectiveResult:
        """
        Evaluate the state, potentially against a baseline, and return an ObjectiveResult.
        """
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class BasePipelineRunner(ABC, BaseModel, Generic[Input_co, MappingState_co, BaselineState_co]):
    """
    Structural interface for the benchmarking pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    config: PipelineOutput = Field(description="Configuration for pipeline execution.")

    @abstractmethod
    def run(self) -> PipelineSummary[Input_co, MappingState_co, BaselineState_co]:
        """Execute the pipeline and return a summary of results."""
        pass

# TODO Put generics into another folder

class PipelineConsumer(ABC, Generic[Input_contra, MappingState_contra, BaselineState_contra]):

    @abstractmethod
    def consume(self, data: PipelineSummary[Input_contra, MappingState_contra, BaselineState_contra]) -> None:
        """
        Consume the pipeline summary data.
        This method must be implemented by subclasses.
        """
        pass

class MappingStateConsumer[MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](ABC):

    @abstractmethod
    def consume(self, state: MAPPING_STATE, baseline: Optional[BASELINE_STATE] = None) -> None:
        """
        Consume the mapping state and optional baseline.
        This method must be implemented by subclasses.
        """
        pass