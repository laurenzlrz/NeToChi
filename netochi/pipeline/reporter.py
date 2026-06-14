from typing import List, Set, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.interfaces import PipelineConsumer
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState
from netochi.pipeline.results import PipelineSummary, ExperimentResult
from netochi.definitions.constants import KEY_GRAPH_TYPE, KEY_UNKNOWN, REPORT_DIVIDER, REPORT_SUBDIVIDER, \
    REPORT_HEADER_BASELINE, REPORT_HEADER_PURE


class SummaryReporter(BaseModel, PipelineConsumer[MappingInput, MappingState[Any, Any], MappingState[Any, Any]]):
    """
    Handles the generation and printing of experiment reports.
    Automatically discovers metrics and formats tables dynamically.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    config: PipelineOutputConfig = Field(description="Configuration for report output formatting and behavior.")

    def consume(self, data: PipelineSummary) -> None:
        self.generate_report_string(data)

    def generate_report_string(self, summary: PipelineSummary) -> None:
        """
        Generates the full summary report as a string.
        """
        if not summary.results:
            self.config.print_console("\nNo results to report.")

        report_lines = []
        all_metric_names = SummaryReporter._discover_metrics(summary.results)
        divider = SummaryReporter._get_divider(all_metric_names)

        # 1. Relative Improvement Table
        report_lines.append(f"\n{divider}")
        report_lines.append(REPORT_HEADER_BASELINE)
        report_lines.append(divider)
        report_lines.append(self._get_table_string(summary.results, all_metric_names, use_raw=False))

        # 2. Absolute Values Table
        report_lines.append(f"\n{divider}")
        report_lines.append(REPORT_HEADER_PURE)
        report_lines.append(divider)
        report_lines.append(self._get_table_string(summary.results, all_metric_names, use_raw=True))

        report_lines.append(divider)
        report_lines.append(f"Total Experiment Time: {summary.total_time_s:.2f}s")
        report_lines.append("Experiment Complete.")

        final_report_str = "\n".join(report_lines)
        self.config.print_console(final_report_str, "summary_report")

    def _get_table_string(self, results: List[ExperimentResult], metric_names: List[str], use_raw: bool) -> str:
        """
        Returns a formatted table for a given set of metrics as a string.
        """
        table_lines = []
        mapper_width, graph_width, metric_width, time_width = 45, 40, 25, 10

        # Format Header
        header = f"{'Mapper':<{mapper_width}} | {'Graph':<{graph_width}}"
        for name in metric_names:
            display_name = SummaryReporter._truncate(name, metric_width)
            header += f" | {display_name:<{metric_width}}"
        header += f" | {'Time (s)':<{time_width}}"

        table_lines.append(header)
        table_lines.append("-" * len(header))

        sorted_results = sorted(
            results,
            key=lambda r: (r.input_metadata.get(KEY_GRAPH_TYPE, KEY_UNKNOWN), r.mapper_name)
        )

        for res in sorted_results:
            mapper_display = SummaryReporter._truncate(res.mapper_name, mapper_width)
            graph_display = SummaryReporter._truncate(res.input_metadata.get(KEY_GRAPH_TYPE, KEY_UNKNOWN), graph_width)

            row = f"{mapper_display:<{mapper_width}} | {graph_display:<{graph_width}}"
            metrics_dict = res.raw_metrics if use_raw else res.metrics

            for name in metric_names:
                val = metrics_dict.get(name, -1.0)
                row += f" | {val:<{metric_width}.2f}"

            row += f" | {res.execution_time_s:<{time_width}.3f}"
            table_lines.append(row)

        return "\n".join(table_lines)

    @staticmethod
    def _get_divider(metric_names: List[str]) -> str:
        """Calculates a divider line based on the number of metrics."""
        # Widths match _print_table: mapper(45) + graph(20) + metrics(N*25) + time(10) + separators
        base_width = 45 + 20 + 10 + (3 * 3) # widths + 3 separators (|)
        metric_width = len(metric_names) * (25 + 3) # width + separator (|)
        return "=" * (base_width + metric_width)

    @staticmethod
    def _discover_metrics(results: List[ExperimentResult]) -> List[str]:
        """
        Collects all unique metric names across all experiment results.
        """
        metrics: Set[str] = set()
        for res in results:
            metrics.update(res.metrics.keys())
            metrics.update(res.raw_metrics.keys())
        # Sort for consistent column order
        return sorted(list(metrics))

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        """Truncates text with ellipsis if it exceeds width."""
        if len(text) > width:
            return text[:width-3] + "..."
        return text

