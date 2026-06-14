from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from netochi.definitions.constants import DEFAULT_METRIC_VALUE
from netochi.mapping.interfaces import MappingState
from netochi.objectives.interfaces import MappingObjective
from netochi.pipeline.interfaces import MappingMetric


class ObjectiveMetric[MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](MappingMetric[MAPPING_STATE, BASELINE_STATE], BaseModel):
    """
    Adapter that shifts Objectives into the Pipeline Metric interface.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    objective: MappingObjective[MAPPING_STATE, BASELINE_STATE] = Field(..., description="The underlying objective to evaluate.")

    def evaluate_against_baseline(self, state: MAPPING_STATE, baseline: Optional[BASELINE_STATE] = None) -> float:
        """
        Evaluate using the objective. 
        If baseline is provided, uses evaluate_against_baseline.
        """
        if baseline is not None:
            return self.objective.evaluate_against_baseline(state, baseline)
        return DEFAULT_METRIC_VALUE

    def evaluate(self, state: MAPPING_STATE) -> float:
        """
        Evaluate using the objective.
        """
        return self.objective.evaluate(state)

    def get_name(self) -> str:
        """
        Use the underlying objective's name for reporting.
        """
        return self.objective.__class__.__name__