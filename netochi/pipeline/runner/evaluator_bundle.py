from typing import List, Optional, Tuple, Dict

from pydantic import BaseModel, ConfigDict, Field

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState, BaseMapper
from netochi.pipeline import MappingMetric
from netochi.pipeline.interfaces import MappingStateConsumer
from netochi.definitions.constants import DEFAULT_METRIC_VALUE



class BaselineStorer[MAPPING_INPUT: MappingInput, BASELINE_STATE: MappingState](BaseMapper[BASELINE_STATE, MAPPING_INPUT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=False)
    _baseline_state: BASELINE_STATE

    @property
    def baseline_state(self) -> Optional[BASELINE_STATE]:
        if (self._baseline_state is not None):
            return self._baseline_state
        return None

    def run(self, mapping_input: MAPPING_INPUT) -> BASELINE_STATE:
        self.precompute_baseline(mapping_input)
        assert self._baseline_state is not None
        return self._baseline_state

    def precompute_baseline(self, mapping_input: MAPPING_INPUT) -> None:
        pass


class EvaluatorBundle[MAPPING_STATE: MappingState, BASELINE_STATE: MappingState, MAPPING_INPUT: MappingInput](BaseModel):
    """Strongly typed container for metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)
    metrics_w_baselines: List[Tuple[MappingMetric[MAPPING_STATE, BASELINE_STATE], BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]] = Field(description="Metrics to evaluate the mapping state against a baseline.")
    hooks: List[Tuple[MappingStateConsumer[MAPPING_STATE, BASELINE_STATE], Optional[BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]]] = Field(description="Hooks to consume the mapping state and baseline after evaluation.")

    def prepare(self, input: MAPPING_INPUT) -> None:
        # Precompute baselines for all metrics
        for metric, baseline_provider in self.metrics_w_baselines:
            baseline_provider.run(input)
        # Precompute baselines for all hooks
        for hook, baseline_provider in self.hooks:
            if baseline_provider is not None:
                baseline_provider.run(input)

    def evaluate_all(self, state: MAPPING_STATE) -> Tuple[Dict[str, float], Dict[str, float]]:
        raw_results: Dict[str, float] = {}
        rel_results: Dict[str, float] = {}
        for hook, baseline_provider in self.hooks:
            baseline = baseline_provider.baseline_state if baseline_provider is not None else None
            hook.consume(state, baseline)
        for metric, baseline_provider in self.metrics_w_baselines:
            baseline = baseline_provider.baseline_state
            # Calculate Raw (Absolute)
            raw_results[metric.get_name()] = metric.evaluate(state)
            # Calculate Relative (if baseline exists)
            if baseline is not None:
                rel_results[metric.get_name()] = metric.evaluate_against_baseline(state, baseline)
            else:
                rel_results[metric.get_name()] = DEFAULT_METRIC_VALUE # No baseline
        return raw_results, rel_results

    @property
    def get_baselines(self) -> List[BaselineStorer[MAPPING_INPUT, BASELINE_STATE]]:
        return [metric_baseline_tuple[1] for metric_baseline_tuple in self.metrics_w_baselines]

