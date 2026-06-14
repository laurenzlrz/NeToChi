
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

from typing import Any
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.interfaces import PipelineConsumer
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState
from netochi.pipeline.results import PipelineSummary


class SummaryArchiver(BaseModel, PipelineConsumer[MappingInput, MappingState[Any, Any], MappingState[Any, Any]]):
    """
    Handles the generation and printing of experiment reports.
    Automatically discovers metrics and formats tables dynamically.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    config: PipelineOutputConfig = Field(description="Configuration for report output formatting and behavior.")

    csv_filename: str = Field(default="results.csv")

    def consume(self, data: PipelineSummary) -> None:
        df = self.create_summary_dataframe(data)
        self.config.save_to_csv(df, self.csv_filename)

    def create_summary_dataframe(self, summary: PipelineSummary) -> pd.DataFrame:
        """
        Flattens pipeline summary results into a structured pandas DataFrame.
        """
        if not summary.results:
            return pd.DataFrame()

        # Collect all possible headers to ensure consistency across rows
        headers = {"mapper", "execution_time_s", "error"}
        for res in summary.results:
            headers.update(res.input_metadata.keys())
            headers.update({f"metric_{k}" for k in res.metrics.keys()})
            headers.update({f"raw_{k}" for k in res.raw_metrics.keys()})

        sorted_headers = sorted(list(headers))
        rows = []

        for res in summary.results:
            # Build the base row dictionary
            row = {
                "mapper": res.mapper_name,
                "execution_time_s": str(res.execution_time_s),
                "error": str(res.error) if res.error else "",
            }
            row.update(res.input_metadata)
            row.update({f"metric_{k}": str(v) for k, v in res.metrics.items()})
            row.update({f"raw_{k}": str(v) for k, v in res.raw_metrics.items()})

            # Ensure all columns exist for this row, defaulting to empty string
            full_row = {h: row.get(h, "") for h in sorted_headers}
            rows.append(full_row)

        # Return the structured DataFrame
        return pd.DataFrame(rows, columns=sorted_headers)
