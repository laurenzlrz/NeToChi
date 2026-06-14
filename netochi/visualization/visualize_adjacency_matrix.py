import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
from typing import Any

from netochi.mapping.interfaces import BaseMosaicMappingState



def plot_sorted_adjacency(graph: gt.Graph, state: BaseMosaicMappingState, filename: str):
    """
    Plots the adjacency matrix of a graph, sorted by core assignment and then local index.
    Draws red boundary lines to separate different cores and labels the axes with Core IDs.
    """
    c = state.c
    x = state.x

    # Sort primarily by core ID (c), secondarily by local ID (x)
    sorted_indices = np.lexsort((x, c))
    adj_matrix: Any = gt.adjacency(graph)
    adj_sorted = adj_matrix.tocsr()[sorted_indices, :][:, sorted_indices]

    plt.figure(figsize=(8, 8))
    plt.spy(adj_sorted.T, markersize=4, color='black')

    sorted_cores = c[sorted_indices]
    core_boundaries = np.where(np.diff(sorted_cores) != 0)[0] + 1

    # Draw red boundary lines
    for boundary in core_boundaries:
        plt.axhline(boundary - 0.5, color='red', linewidth=1.5, alpha=0.6)
        plt.axvline(boundary - 0.5, color='red', linewidth=1.5, alpha=0.6)

    # --- NEW: Calculate and set core labels on axes ---
    tick_positions = []
    tick_labels = []

    start_idx = 0
    # Append total length to boundaries to handle the last core block easily
    all_boundaries = np.append(core_boundaries, len(sorted_cores))

    for boundary in all_boundaries:
        # Calculate the midpoint of the core block to center the label
        midpoint = (start_idx + boundary - 1) / 2.0
        tick_positions.append(midpoint)

        # Grab the actual core ID from the sorted array
        core_id = sorted_cores[start_idx]
        tick_labels.append(f"C{core_id}")

        start_idx = boundary

    # Apply the custom ticks
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)

    # Remove the standard tick marks (the little lines) for a cleaner look
    plt.tick_params(axis='both', which='both', length=0)
    # ---------------------------------------------------

    plt.title("Adjacency Matrix\n(Neurons Grouped by Cores)", pad=20)
    plt.xlabel("Target Neuron")
    plt.ylabel("Source Neuron")

    ax: Any = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


from pydantic import BaseModel, ConfigDict
from netochi.pipeline.interfaces import PipelineConsumer
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.results import PipelineSummary
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import BaseMosaicMappingState

class AdjacencyMatrixVisualizer(BaseModel, PipelineConsumer[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    config: PipelineOutputConfig

    def consume(self, data: PipelineSummary[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]) -> None:
        for res in data.results:
            if res.state is not None:
                graph = res.state.mapping_input.graph
                safe_meta = "_".join(f"{k}-{v}" for k, v in sorted(res.input_metadata.items()))[:50]
                name = f"adjacency_matrix_{res.mapper_name}_{safe_meta}"
                
                # Run the plot logic (without saving to file directly)
                c = res.state.c
                x = res.state.x

                sorted_indices = np.lexsort((x, c))
                adj_matrix: Any = gt.adjacency(graph)
                adj_sorted = adj_matrix.tocsr()[sorted_indices, :][:, sorted_indices]

                plt.figure(figsize=(8, 8))
                plt.spy(adj_sorted.T, markersize=4, color='black')

                sorted_cores = c[sorted_indices]
                core_boundaries = np.where(np.diff(sorted_cores) != 0)[0] + 1

                for boundary in core_boundaries:
                    plt.axhline(boundary - 0.5, color='red', linewidth=1.5, alpha=0.6)
                    plt.axvline(boundary - 0.5, color='red', linewidth=1.5, alpha=0.6)

                tick_positions = []
                tick_labels = []

                start_idx = 0
                all_boundaries = np.append(core_boundaries, len(sorted_cores))

                for boundary in all_boundaries:
                    midpoint = (start_idx + boundary - 1) / 2.0
                    tick_positions.append(midpoint)
                    core_id = sorted_cores[start_idx]
                    tick_labels.append(f"C{core_id}")
                    start_idx = boundary

                plt.xticks(tick_positions, tick_labels)
                plt.yticks(tick_positions, tick_labels)
                plt.tick_params(axis='both', which='both', length=0)

                plt.title("Adjacency Matrix\n(Neurons Grouped by Cores)", pad=20)
                plt.xlabel("Target Neuron")
                plt.ylabel("Source Neuron")

                ax: Any = plt.gca()
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                plt.tight_layout()

                self.config.save_plot(plt, name)
