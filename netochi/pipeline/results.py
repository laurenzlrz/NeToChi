from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState


class ExperimentResult[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel):
    """
    Pydantic model representing the outcome of a single mapper execution on an input.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_id: str = Field(..., description="Unique identifier for the input generated.")
    mapper_name: str = Field(..., description="Name of the mapper/algorithm used.")
    input_metadata: Dict[str, str] = Field(default_factory=dict, description="Metadata about the input graph, e.g., graph type, size.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Relative metrics compared to baseline (e.g., improvement percentages).")
    raw_metrics: Dict[str, float] = Field(default_factory=dict, description="Absolute metric values for the mapping result.")
    execution_time_s: float = Field(0.0, description="Execution time in seconds for this mapping.")
    error: Optional[str] = Field(None, description="Error message if the mapping failed, otherwise None.")
    state: Optional[MAPPING_STATE] = Field(default=None, description="The mapping state object.")
    input: Optional[INPUT] = Field(default=None, description="The input that was mapped.")
    baseline_state: Optional[BASELINE_STATE] = Field(default=None, description="The baseline state object, if applicable.")

class PipelineSummary[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel):
    """
    Aggregated results of a full pipeline run.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    results: List[ExperimentResult[INPUT, MAPPING_STATE, BASELINE_STATE]] = Field(default_factory=list)
    total_time_s: float = Field(0.0, description="Total execution time for the entire pipeline run.")
