from typing import Generic, TypeVar, Optional, Any
from pydantic import ConfigDict

from netochi.pipeline.interfaces import MappingMetric
from netochi.mapping.interfaces import MappingState, MosaicHWMappingState, BaseMosaicMappingState
from netochi.objectives.interfaces import MappingObjective

MAPPING_STATE = TypeVar("MAPPING_STATE", bound=MappingState[Any])
BASELINE_STATE = TypeVar("BASELINE_STATE", bound=MappingState[Any])

class ObjectiveMetric(MappingMetric[MAPPING_STATE, BASELINE_STATE], Generic[MAPPING_STATE, BASELINE_STATE]):
    """
    Adapter that allows any MappingObjective to be used as a Pipeline Metric.
    Supports comparative evaluation if a baseline is provided.
    """
    def __init__(self, objective: MappingObjective[MAPPING_STATE, BASELINE_STATE]) -> None:
        self.objective = objective

    def evaluate_against_baseline(self, state: MAPPING_STATE, baseline: Optional[BASELINE_STATE] = None) -> float:
        """
        Evaluate using the objective. 
        If baseline is provided, uses evaluate_against_baseline.
        """
        if baseline is not None:
            return self.objective.evaluate_against_baseline(state, baseline)
        return -1.0

    def evaluate(self, state: MAPPING_STATE) -> float:
        """
        Evaluate using the objective.
        """
        return self.objective.evaluate(state)


    def get_name(self) -> str:
        """Return the name of the wrapped objective."""
        return self.objective.__class__.__name__

