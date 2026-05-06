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
