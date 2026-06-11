import os

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from netochi.pipeline import PipelineSummary


def plot_results(summary: PipelineSummary):
    """
    Generates a unique bar plot for every combination of input_id and metric.
    Each plot compares all mappers that executed on that specific input.
    """
    # 1. Nest data: input_id -> metric -> mapper_name -> list of values
    # (Using a list handles potential duplicate/replicated runs gracefully)
    output_dir = "result_plots"
    os.makedirs(output_dir, exist_ok=True)

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
            filepath = os.path.join(output_dir, filename)

            # 4. Save file and close the plot context to free up RAM
            plt.savefig(filepath, dpi=300)
            plt.close()  # Prevents runtime warnings about having too many active figures open
