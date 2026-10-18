from typing import Any, List, Optional, Tuple, Dict

from pydantic import BaseModel, ConfigDict, Field

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState, BaseMapper
from netochi.objectives.interfaces import ObjectiveResult
from netochi.pipeline import MappingMetric
from netochi.pipeline.interfaces import MappingStateConsumer
from netochi.pipeline.results import LabeledObjectiveResult
from netochi.definitions.constants import DEFAULT_METRIC_VALUE



class BaselineStorer[MAPPING_INPUT: MappingInput, BASELINE_STATE: MappingState](BaseMapper[BASELINE_STATE, MAPPING_INPUT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=False)
    _baseline_state: BASELINE_STATE

    @property
    def baseline_state(self) -> Optional[BASELINE_STATE]:
        if (self._baseline_state is not None):
            return self._baseline_state
        return None

    def run_mapping(self, mapping_input: MAPPING_INPUT) -> BASELINE_STATE:
        self.precompute_baseline(mapping_input)
        assert self._baseline_state is not None
        return self._baseline_state

    def precompute_baseline(self, mapping_input: MAPPING_INPUT) -> None:
        pass


class EvaluatorBundle[MAPPING_STATE: MappingState, BASELINE_STATE: MappingState, MAPPING_INPUT: MappingInput, OBJECTIVE_LABEL](BaseModel):
    """Strongly typed container for metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)
    # Each entry is (metric, baseline) or (label, metric, baseline); the label is an experiment alias for the objective.
    metrics_w_baselines: List[Tuple[MappingMetric[MAPPING_STATE, BASELINE_STATE], BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]
                              | Tuple[OBJECTIVE_LABEL, MappingMetric[MAPPING_STATE, BASELINE_STATE], BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]] = Field(description="Metrics (optionally labeled) to evaluate the mapping state against a baseline.")
    hooks: List[Tuple[MappingStateConsumer[MAPPING_STATE, BASELINE_STATE], Optional[BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]]] = Field(description="Hooks to consume the mapping state and baseline after evaluation.")

    @property
    def labeled_metrics_w_baselines(self) -> List[Tuple[Optional[OBJECTIVE_LABEL], MappingMetric[MAPPING_STATE, BASELINE_STATE], BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]]:
        return [entry if len(entry) == 3 else (None, *entry) for entry in self.metrics_w_baselines]

    def prepare(self, input: MAPPING_INPUT) -> None:
        # Precompute baselines for all metrics
        for _, metric, baseline_provider in self.labeled_metrics_w_baselines:
            baseline_provider.run(input)
        # Precompute baselines for all hooks
        for hook, baseline_provider in self.hooks:
            if baseline_provider is not None:
                baseline_provider.run(input)

    def evaluate_all(self, state: MAPPING_STATE) -> List[LabeledObjectiveResult[OBJECTIVE_LABEL, ObjectiveResult]]:
        for hook, baseline_provider in self.hooks:
            baseline = baseline_provider.baseline_state if baseline_provider is not None else None
            hook.consume(state, baseline)
            
        results: List[LabeledObjectiveResult[OBJECTIVE_LABEL, ObjectiveResult]] = []
        for label, metric, baseline_provider in self.labeled_metrics_w_baselines:
            baseline = baseline_provider.baseline_state
            res = metric.evaluate(state, baseline)
            results.append(LabeledObjectiveResult(label=label, result=res))
        return results

    @property
    def get_baselines(self) -> List[BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]:
        return [baseline_provider for _, _, baseline_provider in self.labeled_metrics_w_baselines]

