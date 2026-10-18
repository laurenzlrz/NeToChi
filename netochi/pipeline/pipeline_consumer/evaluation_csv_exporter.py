import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from typing import Any
import icontract

from netochi.pipeline.config import PipelineOutput
from netochi.pipeline.interfaces import PipelineConsumer
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState
from netochi.pipeline.results import PipelineSummary
from netochi.result_processing.config import FILENAME_EVALUATION_CSV


def _label_str(label: Any) -> str:
    """Labels are generic; the CSV stores their string form (empty if unset)."""
    return "" if label is None else str(label)


class EvaluationExporterConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    pipeline_output: PipelineOutput = Field(..., description="Pipeline output manager.")
    csv_filename: str = Field(default=FILENAME_EVALUATION_CSV, description="Filename for the evaluation results CSV.")

    def create(self) -> "EvaluationExporter":
        return EvaluationExporter(config=self)


class EvaluationExporter(PipelineConsumer[MappingInput, MappingState[Any, Any], MappingState[Any, Any]]):
    """
    Handles the generation and printing of experiment reports.
    Automatically discovers metrics and formats tables dynamically.
    """

    @icontract.require(lambda config: isinstance(config, EvaluationExporterConfig))
    def __init__(self, config: EvaluationExporterConfig) -> None:
        self.config = config
        self.pipeline_output = config.pipeline_output
        self.csv_filename = config.csv_filename

    def consume(self, data: PipelineSummary) -> None:
        df = self.create_summary_dataframe(data)
        self.pipeline_output.save_to_csv(df, self.csv_filename)


    def create_summary_dataframe(self, summary: PipelineSummary) -> pd.DataFrame:
        """
        Flattens pipeline summary results into a structured pandas DataFrame.
        """
        if not summary.results:
            return pd.DataFrame()

        # Collect all possible headers to ensure consistency across rows
        headers = ["mapper", "input_id", "seed", "metric", "value",
                   "task_label", "input_label", "mapper_label", "objective_label"]
        rows = []


        for res in summary.results:
            # Build the base row dictionary
            for labeled_result in res.objective_results:
                row = {
                    "mapper": res.mapper_name,
                    "input_id": res.input.seed,
                    "seed": res.seed,
                    "metric": labeled_result.result.objective_name,
                    "value": labeled_result.result.value,
                    "task_label": _label_str(res.labels.task),
                    "input_label": _label_str(res.labels.input),
                    "mapper_label": _label_str(res.labels.mapper),
                    "objective_label": _label_str(labeled_result.label)
                }
                full_row = {h: row.get(h, "") for h in headers}
                rows.append(full_row)

        # Return the structured DataFrame
        return pd.DataFrame(rows, columns=headers)
