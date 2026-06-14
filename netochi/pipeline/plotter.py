import matplotlib # type: ignore[import-untyped]

from netochi.pipeline.PipelineConsumer import PipelineConsumer

matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt # type: ignore[import-untyped]
import numpy as np
from pathlib import Path
from typing import Dict, Set, List
from pydantic import BaseModel, ConfigDict
from netochi.pipeline.results import PipelineSummary
from netochi.pipeline.config import PipelineOutputConfig

class PipelinePlotter(PipelineConsumer):
    """
    Generates interpretable plots from pipeline results.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    config: PipelineOutputConfig

    def consume(self, data: PipelineSummary) -> None:
        self.plot_all(data)

    def plot_all(self, summary: PipelineSummary) -> None:
        """
        Generates all configured plots (raw and relative) and saves them.
        """
        if not summary.results:
            return
        
        # Identify all available relative metrics
        rel_metrics: Set[str] = set()
        raw_metrics: Set[str] = set()
        for res in summary.results:
            rel_metrics.update(res.metrics.keys())
            raw_metrics.update(res.raw_metrics.keys())
            
        # Generate relative plots
        for metric in rel_metrics:
            self._plot_metric_comparison(summary, metric, is_raw=False)
        
        # Generate absolute plots
        for metric in raw_metrics:
            self._plot_metric_comparison(summary, metric, is_raw=True)
            
        # Generate execution time comparison
        self._plot_execution_times(summary)
        
        # Generate per-network individual plots
        self._plot_individual_networks(summary)

    def _plot_metric_comparison(self, summary: PipelineSummary, metric: str, is_raw: bool = False) -> None:
        # Organize data: graph_type -> mapper -> value
        data: Dict[str, Dict[str, float]] = {}
        mappers: Set[str] = set()
        
        for res in summary.results:
            source = res.raw_metrics if is_raw else res.metrics
            if metric not in source:
                continue
            graph_type = res.input_metadata.get("graph_type", "Unknown")
            mapper = res.mapper_name
            mappers.add(mapper)
            
            if graph_type not in data:
                data[graph_type] = {}
            data[graph_type][mapper] = source[metric]
            
        if not data:
            return
            
        graph_types = sorted(list(data.keys()))
        sorted_mappers = sorted(list(mappers))
        
        # Plotting logic
        x = np.arange(len(graph_types))
        width = 0.8 / len(sorted_mappers)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for i, mapper in enumerate(sorted_mappers):
            values = [data[gt].get(mapper, 0.0) for gt in graph_types]
            offset = (i - (len(sorted_mappers) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=mapper, color=self.config.palette[i % len(self.config.palette)])
            
        ax.set_ylabel("Metric Value")
        suffix = " (Absolute)" if is_raw else " (Relative to Baseline)"
        ax.set_title(f"Comparison: {metric}{suffix}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(graph_types, rotation=15)
        ax.legend(title="Mappers", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        suffix_file = "raw" if is_raw else "rel"
        filename = self.config.plot_filename_pattern.format(metric=metric) + f"_{suffix_file}"
        self.config.save_plot(plt, filename)

    def _plot_execution_times(self, summary: PipelineSummary) -> None:
        # Organize data: mapper -> average execution time
        times: Dict[str, List[float]] = {}
        for res in summary.results:
            if res.mapper_name not in times:
                times[res.mapper_name] = []
            times[res.mapper_name].append(res.execution_time_s)
            
        if not times:
            return
            
        sorted_mappers = sorted(list(times.keys()))
        avg_times = [np.mean(times[m]) for m in sorted_mappers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(sorted_mappers, avg_times, color=self.config.palette[0])
        
        ax.set_xlabel("Average Execution Time (s)")
        ax.set_title("Mapper Performance (Lower is Faster)", fontsize=14, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f' {width:.2f}s', va='center')
            
        plt.tight_layout()
        self.config.save_plot(plt, "execution_times")

    def _plot_individual_networks(self, summary: PipelineSummary) -> None:
        """
        Generates a separate plot for every unique network configuration.
        """
        # Group by unique metadata string
        network_data: Dict[str, List[ExperimentResult]] = {}
        for res in summary.results:
            # Create a stable identifier from metadata excluding dynamic values if any
            meta = res.input_metadata
            ident = " | ".join([f"{k}:{v}" for k, v in sorted(meta.items())])
            if ident not in network_data:
                network_data[ident] = []
            network_data[ident].append(res)
        
        for ident, results in network_data.items():
            # For each metric, plot mappers for this specific network
            all_metrics: Set[str] = set()
            for r in results:
                all_metrics.update(r.raw_metrics.keys())
            
            for metric in all_metrics:
                mappers = [r.mapper_name for r in results]
                values = [r.raw_metrics.get(metric, 0.0) for r in results]
                
                if not values: continue
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(mappers, values, color=self.config.palette[:len(mappers)])
                
                ax.set_ylabel(metric)
                ax.set_title(f"Network: {ident}\nMetric: {metric}", fontsize=10)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                safe_ident = ident.replace(" ", "_").replace(":", "-").replace("|", "_")[:50]
                filename = f"net_{safe_ident}_{metric}"
                self.config.save_plot(plt, filename)
