from typing import Iterator, Tuple, Dict, Any, List
import graph_tool.all as gt
import networkx as nx
from netochi.pipeline.core import BaseInputFactory, MappingInput, FixedHardwareInput
from netochi.network_generator.random_generator import RandomGeneratorConfig, RandomNetwork
from netochi.mapping.hardware_config import HardwareConfig

def nx_to_gt(nx_g: nx.DiGraph) -> gt.Graph:
    """Helper to convert networkx graph to graph-tool graph."""
    gt_g = gt.Graph(directed=nx_g.is_directed())
    gt_g.add_vertex(nx_g.number_of_nodes())
    gt_g.add_edge_list(list(nx_g.edges()))
    return gt_g

class RandomNetworkFactory(BaseInputFactory):
    """Factory generating synthetic networks for a fixed hardware configuration."""
    
    def __init__(self, hw_config: HardwareConfig, probabilities: List[float], seed: int = 42):
        """Initialize with hardware config, edge probabilities, and optional seed."""
        self.hw_config = hw_config
        self.probabilities = probabilities
        self.seed = seed
        
    def generate(self) -> Iterator[Tuple[MappingInput, Dict[str, Any]]]:
        for p in self.probabilities:
            net_config = RandomGeneratorConfig(
                nodes_per_router=self.hw_config.nodes_per_router,
                neurons_per_core=self.hw_config.neurons_per_core,
                router_levels=self.hw_config.router_levels,
                edge_probability=p,
                seed=self.seed
            )
            
            generator = RandomNetwork(net_config)
            generator.generate()
            nx_graph = generator._graph
            graph = nx_to_gt(nx_graph)
            
            meta = {
                "graph_type": "RandomNetwork",
                "edge_prob": p,
                "nodes": graph.num_vertices(),
                "edges": graph.num_edges(),
                "ground_truth_assignments": generator._assignment
            }
            
            input_pair = FixedHardwareInput(graph=graph, hw_config=self.hw_config, metadata=meta)
            
            # We can optionally attach the ground truth to the graph or return it if needed,
            # but for now we just return the input pair and metadata.
            yield input_pair, meta
