from typing import List, Set, Dict, Any
from netochi.pipeline.results import PipelineSummary, ExperimentResult
from netochi.pipeline.constants import (
    REPORT_DIVIDER, REPORT_SUBDIVIDER, REPORT_HEADER_BASELINE, REPORT_HEADER_PURE,
    KEY_GRAPH_TYPE, KEY_UNKNOWN
)

class SummaryReporter:
    """
    Handles the generation and printing of experiment reports.
    Automatically discovers metrics and formats tables dynamically.
    """

    @staticmethod
    def print_report(summary: PipelineSummary) -> None:
        """
        Main entry point for printing the summary report.
        """
        if not summary.results:
            print("\nNo results to report.")
            return

        # 1. Discover all metrics present in the results
        all_metric_names = SummaryReporter._discover_metrics(summary.results)
        
        # 2. Print Relative Improvement Table
        print("\n" + SummaryReporter._get_divider(all_metric_names))
        print(REPORT_HEADER_BASELINE)
        print(SummaryReporter._get_divider(all_metric_names))
        SummaryReporter._print_table(summary.results, all_metric_names, use_raw=False)

        # 3. Print Absolute Values Table
        print("\n" + SummaryReporter._get_divider(all_metric_names))
        print(REPORT_HEADER_PURE)
        print(SummaryReporter._get_divider(all_metric_names))
        SummaryReporter._print_table(summary.results, all_metric_names, use_raw=True)

        print(SummaryReporter._get_divider(all_metric_names))
        print(f"Total Experiment Time: {summary.total_time_s:.2f}s")
        print("Experiment Complete.")

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
    def _print_table(results: List[ExperimentResult], metric_names: List[str], use_raw: bool) -> None:
        """
        Prints a formatted table for a given set of metrics.
        """
        # Header setup
        mapper_width = 45
        graph_width = 20
        metric_width = 25
        time_width = 10
        
        # Format Header
        header = f"{'Mapper':<{mapper_width}} | {'Graph':<{graph_width}}"
        for name in metric_names:
            # Shorten name if too long for column
            display_name = (name[:metric_width-3] + '..') if len(name) > metric_width else name
            header += f" | {display_name:<{metric_width}}"
        header += f" | {'Time (s)':<{time_width}}"
        
        print(header)
        print("-" * len(header))

        # Sort results by graph type then mapper name
        sorted_results = sorted(
            results, 
            key=lambda r: (r.input_metadata.get(KEY_GRAPH_TYPE, KEY_UNKNOWN), r.mapper_name)
        )

        for res in sorted_results:
            row = f"{res.mapper_name:<{mapper_width}} | {res.input_metadata.get(KEY_GRAPH_TYPE, KEY_UNKNOWN):<{graph_width}}"
            
            metrics_dict = res.raw_metrics if use_raw else res.metrics
            
            for name in metric_names:
                val = metrics_dict.get(name, -1.0)
                # Format based on value type or name (heuristic)
                if "Percentage" in name or "Incons" in name:
                    row += f" | {val:<{metric_width}.2f}"
                else:
                    row += f" | {val:<{metric_width}.2f}"
            
            row += f" | {res.execution_time_s:<{time_width}.3f}"
            print(row)
