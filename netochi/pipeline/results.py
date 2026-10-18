from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState
from netochi.definitions.constants import DEFAULT_METRIC_VALUE
from netochi.objectives.interfaces import ObjectiveResult

class LabeledObjectiveResult[OBJECTIVE_LABEL, RESULT: ObjectiveResult](BaseModel):
    """
    Links an objective result to the experiment label of the metric that produced it.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    label: Optional[OBJECTIVE_LABEL] = Field(default=None, description="Experiment label of the objective, None if unlabeled.")
    result: RESULT = Field(..., description="The objective result.")


class ExperimentLabels[TASK_LABEL, INPUT_LABEL, MAPPER_LABEL](BaseModel):
    """
    Experiment labels (aliases) attached by the pipeline, independent of the content-based names.
    Any label that was not set is None.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    task: Optional[TASK_LABEL] = Field(default=None, description="Label of the ExperimentTaskBase.")
    input: Optional[INPUT_LABEL] = Field(default=None, description="Label of the input generator.")
    mapper: Optional[MAPPER_LABEL] = Field(default=None, description="Label of the mapper (ExperimentTaskRun).")


class ExperimentResult[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState, RESULT: ObjectiveResult](BaseModel):
    """
    Pydantic model representing the outcome of a single mapper execution on an input.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_id: str = Field(..., description="Unique identifier for the input generated.")
    seed: int = Field(..., description="The seed used to generate the input.")
    mapper_name: str = Field(..., description="Name of the mapper/algorithm used.")
    input_metadata: Dict[str, str] = Field(default_factory=dict, description="Metadata about the input graph, e.g., graph type, size.")
    objective_results: List[LabeledObjectiveResult[Any, RESULT]] = Field(default_factory=list, description="Objective results of the evaluation, each with its experiment label.")
    input: Optional[INPUT] = Field(description="The input that was mapped.")
    error: Optional[str] = Field(None, description="Error message if the mapping failed, otherwise None.")
    state: Optional[MAPPING_STATE] = Field(default=None, description="The mapping state object.")
    baseline_state: Optional[BASELINE_STATE] = Field(default=None, description="The baseline state object, if applicable.")
    labels: ExperimentLabels[Any, Any, Any] = Field(default_factory=ExperimentLabels, description="Experiment labels of task, input and mapper.")

    @property
    def raw_metrics(self) -> Dict[str, float]:
        return {r.result.objective_name: r.result.value for r in self.objective_results}

    @property
    def metrics(self) -> Dict[str, float]:
        return {
            r.result.objective_name: (r.result.baseline_value if r.result.baseline_value is not None else DEFAULT_METRIC_VALUE)
            for r in self.objective_results
        }

class PipelineSummary[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel):
    """
    Aggregated results of a full pipeline run.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    results: List[ExperimentResult[INPUT, MAPPING_STATE, BASELINE_STATE, Any]] = Field(default_factory=list)
    total_time_s: float = Field(0.0, description="Total execution time for the entire pipeline run.")
