import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt


def plot_clustering_comparison(
        g: gt.Graph,
        initial_assignment: np.ndarray,
        inferred_assignment: np.ndarray,
        filename: str,
        cluster_spread: float = 1.5
):
    """
    Visualizes graph clustering performance.
    - Spatial position represents the INITIAL (ground truth) cluster.
    - Node color represents the INFERRED cluster.
    """
    # 1. Initialize property maps for position and color
    pos = g.new_vertex_property("vector<double>")
    v_color = g.new_vertex_property("vector<double>")

    unique_initial = np.unique(initial_assignment)
    unique_inferred = np.unique(inferred_assignment)

    # 2. Determine centers for initial clusters (arrange them in a large circle)
    num_initial = len(unique_initial)
    # Scale radius based on number of clusters so they don't overlap
    radius = max(2.0, num_initial * cluster_spread * 0.3) # change scaling and max radius so update distance between clusters

    centers = {}
    for i, c_id in enumerate(unique_initial):
        theta = 2 * np.pi * i / num_initial
        centers[c_id] = (radius * np.cos(theta), radius * np.sin(theta))

    # 3. Assign node positions based on initial assignment
    for v in g.vertices():
        v_idx = int(v)
        c_init = initial_assignment[v_idx]
        cx, cy = centers[c_init]

        # Sample a random position within a circle of radius 'cluster_spread'
        r = cluster_spread * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()

        pos[v] = [cx + r * np.cos(theta), cy + r * np.sin(theta)]

    # 4. Assign node colors based on inferred assignment
    # Get a colormap from matplotlib (tab20 supports up to 20 distinct colors nicely)
    cmap = plt.get_cmap('tab20')
    inferred_to_idx = {c_id: i for i, c_id in enumerate(unique_inferred)}

    for v in g.vertices():
        v_idx = int(v)
        c_inf = inferred_assignment[v_idx]
        color_idx = inferred_to_idx[c_inf]

        # cmap returns an RGBA tuple, which graph-tool accepts natively
        v_color[v] = cmap(color_idx / max(1, len(unique_inferred) - 1))

    # 5. Draw and save the graph
    gt.graph_draw(
        g,
        pos=pos,
        vertex_fill_color=v_color,
        vertex_color="black",  # Border color for nodes
        vertex_size=8,
        edge_pen_width=2.5,
        edge_color=[0.5, 0.5, 0.5, 0.2],  # Semi-transparent gray edges to reduce visual clutter
        output=filename,
        output_size=(1000, 1000),
        fit_view=True
    )


from netochi.pipeline.interfaces import PipelineConsumer
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.results import PipelineSummary
from netochi.mapping.interfaces import BaseMosaicMappingState
from netochi.input_generator.interfaces import MosaicMappingInput
from typing import Any

class ClusteringVisualizer(PipelineConsumer[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]):
    config: PipelineOutputConfig

    def consume(self, data: PipelineSummary[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]) -> None:
        for res in data.results:
            if res.state is not None and res.state.mapping_input.assignment is not None:
                graph = res.state.mapping_input.graph
                initial_assignment = res.state.mapping_input.assignment.neuron_core_pre_assignment
                inferred_assignment = res.state.c
                
                safe_meta = "_".join(f"{k}-{v}" for k, v in sorted(res.input_metadata.items()))[:50]
                name = f"clustering_comparison_{res.mapper_name}_{safe_meta}"
                
                class GraphToolPlotWrapper:
                    def savefig(self, save_path, dpi=150):
                        plot_clustering_comparison(
                            graph,
                            initial_assignment=initial_assignment,
                            inferred_assignment=inferred_assignment,
                            filename=str(save_path)
                        )
                    def close(self):
                        pass

                self.config.save_plot(GraphToolPlotWrapper(), name)
