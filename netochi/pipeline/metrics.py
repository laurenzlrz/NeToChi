from typing import Generic, TypeVar, Any
from pydantic import ConfigDict

from netochi.pipeline.interfaces import MappingMetric
from netochi.mapping.interfaces import MappingState
from netochi.objectives.interfaces import MappingObjective

MAPPING_STATE = TypeVar("MAPPING_STATE", bound=MappingState)
BASELINE_STATE = TypeVar("BASELINE_STATE", bound=MappingState)

class ObjectiveMetric(MappingMetric[MAPPING_STATE], Generic[MAPPING_STATE, BASELINE_STATE]):
    """
    Adapter that allows any MappingObjective to be used as a Pipeline Metric.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    objective: MappingObjective[MAPPING_STATE, BASELINE_STATE]

    def evaluate(self, state: MAPPING_STATE) -> float:
        """Evaluate using the internal objective's direct evaluate method."""
        return self.objective.evaluate(state)

    @classmethod
    def get_name(cls) -> str:
        return "ObjectiveMetric"
