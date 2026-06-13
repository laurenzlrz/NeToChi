from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ExperimentResult(BaseModel):
    """
    Pydantic model representing the outcome of a single mapper execution on an input.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mapper_name: str = Field(..., description="Name of the mapper/algorithm used.")
    input_metadata: Dict[str, str] = Field(default_factory=dict, description="Metadata about the input graph, e.g., graph type, size.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Relative metrics compared to baseline (e.g., improvement percentages).")
    raw_metrics: Dict[str, float] = Field(default_factory=dict, description="Absolute metric values for the mapping result.")
    execution_time_s: float = Field(0.0, description="Execution time in seconds for this mapping.")
    error: Optional[str] = Field(None, description="Error message if the mapping failed, otherwise None.")


class PipelineSummary(BaseModel):
    """
    Aggregated results of a full pipeline run.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    results: List[ExperimentResult] = Field(default_factory=list)
    total_time_s: float = Field(0.0, description="Total execution time for the entire pipeline run.")
