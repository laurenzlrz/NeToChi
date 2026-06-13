from typing import List, Optional, Tuple, Dict

from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import MappingState
from netochi.pipeline import MappingMetric
from netochi.definitions.constants import DEFAULT_METRIC_VALUE


class EvaluatorBundle[MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel):
    """Strongly typed container for metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)
    metrics: List[MappingMetric[MAPPING_STATE, BASELINE_STATE]] = Field(description="Metrics to evaluate the mapping state against a baseline.")

    def evaluate_all(self, state: MAPPING_STATE, baseline: Optional[BASELINE_STATE]) -> Tuple[Dict[str, float], Dict[str, float]]:
        raw_results: Dict[str, float] = {}
        rel_results: Dict[str, float] = {}
        for metric in self.metrics:
            # Calculate Raw (Absolute)
            raw_results[metric.get_name()] = metric.evaluate(state)
            # Calculate Relative (if baseline exists)
            if baseline is not None:
                rel_results[metric.get_name()] = metric.evaluate_against_baseline(state, baseline)
            else:
                rel_results[metric.get_name()] = DEFAULT_METRIC_VALUE # No baseline
        return raw_results, rel_results
