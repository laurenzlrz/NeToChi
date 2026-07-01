from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import icontract

from netochi.definitions.constants import DEFAULT_METRIC_VALUE
from netochi.mapping.interfaces import MappingState
from netochi.objectives.interfaces import MappingObjective, AbstractObjectiveConfig
from netochi.pipeline.interfaces import MappingMetric


class ObjectiveMetricConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    objective_config: AbstractObjectiveConfig = Field(..., description="The configuration for the objective to evaluate.")

    def create(self) -> "ObjectiveMetric":
        objective = self.objective_config.create()
        return ObjectiveMetric(config=self, objective=objective)


class ObjectiveMetric[MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](MappingMetric[MAPPING_STATE, BASELINE_STATE]):
    """
    Adapter that shifts Objectives into the Pipeline Metric interface.
    """

    @icontract.require(lambda config, objective: isinstance(config, ObjectiveMetricConfig) and isinstance(objective, MappingObjective))
    def __init__(self, config: ObjectiveMetricConfig, objective: MappingObjective[Any, Any]) -> None:
        self.config = config
        self.objective = objective

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
        """Return the name of the wrapped objective."""
        return self.objective.get_name()
