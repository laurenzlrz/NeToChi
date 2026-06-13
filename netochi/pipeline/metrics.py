from netochi.definitions.constants import DEFAULT_METRIC_VALUE
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import MappingState, HWNetworkMappingState, MosaicHWMappingState, BaseMosaicMappingState
from netochi.objectives.interfaces import MappingObjective
from typing import Generic, TypeVar, Optional, Any
from pydantic import ConfigDict, BaseModel

from netochi.pipeline.interfaces import MappingMetric


class ObjectiveMetric[MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel, MappingMetric[MAPPING_STATE, BASELINE_STATE]):
    """
    Adapter that shifts Objectives into the Pipeline Metric interface.
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


class InconsistencyPercentageMetric(MappingMetric[BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]):
    """
    Evaluates an inconsistency-based objective and returns it as a percentage of total edges.
    """
    def __init__(self, objective: MappingObjective[BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]) -> None:
        self.objective = objective
    
    def evaluate(self, state: BaseMosaicMappingState[MosaicMappingInput[Any]]) -> float:
        invalid_edges = self.objective.evaluate(state)
        total_edges = state.mapping_input.graph.num_edges()

        if total_edges == 0:
            return 0.0
        return (float(invalid_edges) / float(total_edges)) * 100.0

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[MosaicMappingInput[Any]], baseline: Optional[BaseMosaicMappingState[MosaicMappingInput[Any]]] = None) -> float:
        if baseline is not None:
            baseline_val = self.evaluate(baseline)
            if baseline_val == 0:
                return 1.0 if self.evaluate(state) == 0 else 100.0 # Or some other convention
            return self.evaluate(state) / baseline_val
        return -1.0

    def get_name(self) -> str:
        return "InconsistencyPercentage"

