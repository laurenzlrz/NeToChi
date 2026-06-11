from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ExperimentResult(BaseModel):
    """
    Pydantic model representing the outcome of a single mapper execution on an input.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mapper_name: str
    input_id: str
    input_metadata: Dict[str, str]
    metrics: Dict[str, float]
    raw_metrics: Dict[str, float] = Field(default_factory=dict)
    execution_time_s: float
    error: Optional[str] = None


class PipelineSummary(BaseModel):
    """
    Aggregated results of a full pipeline run.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    results: List[ExperimentResult] = Field(default_factory=list)
    total_time_s: float = 0.0
