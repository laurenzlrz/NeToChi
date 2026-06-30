import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import pandas as pd

from netochi.definitions.constants import NAME_OBJ_EXECUTION_TIME
from netochi.pipeline import PipelineSummary
from netochi.pipeline.config import find_repo_root
from netochi.result_processing.config import OUTPUT_DIR_PLOTS, FILENAME_EVALUATION_CSV, OUTPUT_DIR_REL_RUN_PLOTS, \
    RUN_PREFIX, OUTPUT_DIR_RESULTS


def plot_results_summary(summary: PipelineSummary):
    """
    Generates a unique bar plot for every combination of input_id and metric.
    Each plot compares all mappers that executed on that specific input.
    """
    # 1. Nest data: input_id -> metric -> mapper_name -> list of values
    # (Using a list handles potential duplicate/replicated runs gracefully)
    os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)

    structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for res in summary.results:
        # Skip failed runs
        if res.error is not None:
            continue

        input_id = res.input_id
        mapper = res.mapper_name

        # Populating using raw_metrics (switch to res.metrics if you want normalized values)
        for metric, value in res.raw_metrics.items():
            structured_data[input_id][metric][mapper].append(value)

    # 2. Iterate and generate plots sequentially
    for input_id, metrics_dict in structured_data.items():
        for metric, mappers_dict in metrics_dict.items():

            mappers = list(mappers_dict.keys())
            # Calculate mean value for the mapper on this specific input
            values = [np.mean(mappers_dict[m]) for m in mappers]

            # Initialize a fresh figure
            plt.figure(figsize=(10, 6))

            # Create the bar chart
            bars = plt.bar(mappers, values, color='seagreen', edgecolor='black')

            # Labels and titles
            plt.title(f"Input: {input_id}\nPerformance Comparison: {metric}", fontsize=13, pad=15)
            plt.ylabel(f"{metric}", fontsize=11)

            # Dynamic label handling
            plt.xticks(rotation=35, ha='right')

            # Set structural limits to leave room for text labels on top of the bars
            if values:
                max_val = max(values)
                # Safeguard against all zeros
                plt.ylim(0, max_val * 1.15 if max_val > 0 else 1.0)

                # Add data values on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )

            plt.tight_layout()

            clean_input_id = str(input_id).replace(" ", "_").replace("/", "_").replace("\\", "_")
            clean_metric = str(metric).replace(" ", "_").replace("/", "_").replace("\\", "_")

            filename = f"{clean_input_id}_{clean_metric}.png"
            filepath = os.path.join(OUTPUT_DIR_PLOTS, filename)

            # 4. Save file and close the plot context to free up RAM
            plt.savefig(filepath, dpi=300)
            plt.close()  # Prevents runtime warnings about having too many active figures open


def plot_results_csv(run_id: str, metrics: List[str], mappers: List[str] | None, inputs: List[str]):
    run_path = find_repo_root() / OUTPUT_DIR_RESULTS / f"{RUN_PREFIX}{run_id}"
    csv_path = run_path / f"{FILENAME_EVALUATION_CSV}.csv"
    output_path = run_path / OUTPUT_DIR_REL_RUN_PLOTS
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for metric in metrics:
        for input_id in inputs:

            # Filter the dataframe for the current metric, input_id, and requested mappers
            mask = (
                    (df['metric'] == metric) &
                    (df['input_id'] == input_id)
            )
            if mappers is not None:
                mask = mask & df['mapper'].isin(mappers)

            plot_data = df[mask]

            # Skip if there's no data for this combination
            if plot_data.empty:
                print(f"No data found for metric '{metric}' and input '{input_id}'. Skipping...")
                continue

            # Create the plot
            plt.figure(figsize=(10, 6))

            # Plot bars mapping the 'mapper' to the 'value' column
            plt.bar(
                plot_data['mapper'],
                plot_data['value'],
                color='steelblue',
                edgecolor='black',
                zorder=2
            )

            # Formatting and styling
            plt.title(f"{metric} for {input_id}", fontsize=14, pad=15)
            plt.xlabel("Mapper", fontsize=12)
            plt.ylabel(metric, fontsize=12)

            # Rotate x-axis labels in case mapper names are long
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
            plt.tight_layout()

            # Create a safe filename (remove spaces, parentheses, and equal signs)
            safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "")
            safe_input_id = input_id.replace("=", "_").replace(",", "_")
            file_name = f"{safe_metric}_{safe_input_id}.png"

            # Save and close
            plt.savefig(output_path / file_name, dpi=300)
            plt.close()

    print(f"Successfully generated plots in {output_path}")

if __name__ == "__main__":
    plot_results_csv("020", metrics=["Inconsistencies", NAME_OBJ_EXECUTION_TIME], mappers=None, inputs=["Mosaic_R=3_l=3_N=20_p=0.5_seed=42", "ErdosRenyi_n=60_p=0.1_seed=42"])
