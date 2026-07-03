from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState
from netochi.definitions.constants import DEFAULT_METRIC_VALUE
from netochi.objectives.interfaces import ObjectiveResult

class ExperimentResult[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState, RESULT: ObjectiveResult](BaseModel):
    """
    Pydantic model representing the outcome of a single mapper execution on an input.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_id: str = Field(..., description="Unique identifier for the input generated.")
    mapper_name: str = Field(..., description="Name of the mapper/algorithm used.")
    input_metadata: Dict[str, str] = Field(default_factory=dict, description="Metadata about the input graph, e.g., graph type, size.")
    objective_results: List[RESULT] = Field(default_factory=list, description="Objective results of the evaluation.")
    error: Optional[str] = Field(None, description="Error message if the mapping failed, otherwise None.")
    state: Optional[MAPPING_STATE] = Field(default=None, description="The mapping state object.")
    input: Optional[INPUT] = Field(default=None, description="The input that was mapped.")
    baseline_state: Optional[BASELINE_STATE] = Field(default=None, description="The baseline state object, if applicable.")

    @property
    def raw_metrics(self) -> Dict[str, float]:
        return {r.objective_name: r.value for r in self.objective_results}

    @property
    def metrics(self) -> Dict[str, float]:
        return {
            r.objective_name: (r.baseline_value if r.baseline_value is not None else DEFAULT_METRIC_VALUE)
            for r in self.objective_results
        }

class PipelineSummary[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel):
    """
    Aggregated results of a full pipeline run.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    results: List[ExperimentResult[INPUT, MAPPING_STATE, BASELINE_STATE, Any]] = Field(default_factory=list)
    total_time_s: float = Field(0.0, description="Total execution time for the entire pipeline run.")
