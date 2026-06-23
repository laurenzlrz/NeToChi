import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig



def plot_routing_hierarchy(hw_config: MosaicHardwareConfig, filename: Optional[str] = None):
    """
    Plots the routing hierarchy tree for a given MosaicHardwareConfig.
    Leaves (cores) are circles, routers are rectangles.
    """
    G = nx.Graph()
    pos = {}
    labels = {}

    # Track nodes by their level to calculate coordinates layer by layer
    nodes_by_level = {0: [f"C{i}" for i in range(hw_config.total_cores)]}

    # 1. Initialize the leaves (Cores) at y=0
    for i, core_id in enumerate(nodes_by_level[0]):
        pos[core_id] = (i, 0)
        labels[core_id] = core_id

    # 2. Build the tree level by level up to the root
    for level in range(1, hw_config.router_levels + 1):
        nodes_by_level[level] = []
        children = nodes_by_level[level - 1]

        # Group children based on the router fan-out
        for i in range(0, len(children), hw_config.nodes_per_router):
            group = children[i: i + hw_config.nodes_per_router]
            router_idx = len(nodes_by_level[level])

            # Unique internal ID for networkx
            router_id = f"Router_L{level}_{router_idx}"
            nodes_by_level[level].append(router_id)

            # Display label requested: R1, R2, ..., R_max_level
            labels[router_id] = f"R{level}"

            # The X position of the router is the average of its children's X positions
            avg_x = sum(pos[child][0] for child in group) / len(group)
            pos[router_id] = (avg_x, level)

            # Add edges between this router and its children
            for child in group:
                G.add_edge(router_id, child)

    # 3. Separate nodes by type for distinct shape plotting
    cores = nodes_by_level[0]
    routers = [node for level in range(1, hw_config.router_levels + 1) for node in nodes_by_level[level]]

    # 4. Plotting setup (dynamically scale width based on total cores)
    fig_width = max(8, hw_config.total_cores * 0.6)
    fig_height = max(5, hw_config.router_levels * 1.5)
    plt.figure(figsize=(fig_width, fig_height))

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5)

    # Draw cores (Leaves = circles 'o')
    nx.draw_networkx_nodes(
        G, pos, nodelist=cores, node_shape='o',
        node_color='white', edgecolors='black', node_size=800
    )

    # Draw routers (Internal nodes = squares 's')
    nx.draw_networkx_nodes(
        G, pos, nodelist=routers, node_shape='s',
        node_color='white', edgecolors='black', node_size=800
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")

    # Final plot formatting
    plt.title(
        f"Routing Hierarchy\n(Cores: {hw_config.total_cores}, Levels: {hw_config.router_levels}, Children per Router: {hw_config.nodes_per_router})",
        fontsize=14, pad=20
    )
    # Give a little padding so border nodes aren't cut off
    plt.margins(x=0.05, y=0.1)
    plt.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


from pydantic import BaseModel, Field, ConfigDict
import icontract
from netochi.pipeline.interfaces import PipelineConsumer
from netochi.pipeline.config import PipelineOutput
from netochi.pipeline.results import PipelineSummary
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import BaseMosaicMappingState


class RoutingHierarchyVisualizerConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    pipeline_output: PipelineOutput = Field(..., description="Pipeline output manager.")

    def create(self) -> "RoutingHierarchyVisualizer":
        return RoutingHierarchyVisualizer(config=self)


class RoutingHierarchyVisualizer(PipelineConsumer[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]):

    @icontract.require(lambda config: isinstance(config, RoutingHierarchyVisualizerConfig))
    def __init__(self, config: RoutingHierarchyVisualizerConfig) -> None:
        self.config = config
        self.pipeline_output = config.pipeline_output

    def consume(self, data: PipelineSummary[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]) -> None:
        seen_hw_configs = set()
        for res in data.results:
            if res.state is not None:
                hw_config = res.state.hw_to_evaluate
                hw_key = (hw_config.nodes_per_router, hw_config.neurons_per_core, hw_config.router_levels, hw_config.slice_factor)
                if hw_key not in seen_hw_configs:
                    seen_hw_configs.add(hw_key)
                    name = f"routing_hierarchy_{hw_config.nodes_per_router}_{hw_config.neurons_per_core}_{hw_config.router_levels}"
                    
                    plot_routing_hierarchy(hw_config)
                    self.pipeline_output.save_plot(plt, name)



