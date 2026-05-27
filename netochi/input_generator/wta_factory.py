from typing import Dict, Any, Optional
import networkx as nx
import numpy as np
import graph_tool.all as gt
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from netochi.input_generator.interfaces import BaseInputFactory, MosaicMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt

class WTAFactory(BaseModel, HWBaseInputFactory[MosaicMappingInput]):
    """
    Factory generating Winner-Takes-All (WTA) networks using a hub-and-spoke skeleton.
    - Excitatory Pool: n-1 nodes.
    - Inhibitory Hub: 1 node (the last node).
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    hw_config: MosaicHardwareConfig
    n: int = Field(..., ge=2, description="Total number of neurons (nodes).")
    probability: float = Field(..., ge=0, le=1, description="Connectivity probability.")
    seed: int = 42
    
    _graph: Optional[nx.DiGraph] = PrivateAttr(default=None)

    def generate(self) -> MosaicHWMappingInput[Any]:
        """Generate a single MosaicMappingInput with a WTA graph."""
        rng = np.random.default_rng(self.seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n))
        
        excitatory_nodes = range(self.n - 1)
        inhibitory_hub = self.n - 1
        
        edges = []
        for e_node in excitatory_nodes:
            # 1. Feedback: Competitor to Referee
            if rng.random() < self.probability:
                edges.append((e_node, inhibitory_hub))
            
            # 2. Inhibition: Referee to Competitor
            if rng.random() < self.probability:
                edges.append((inhibitory_hub, e_node))
        
        graph.add_edges_from(edges)
        object.__setattr__(self, '_graph', graph)
        
        gt_graph = nx_to_gt(graph)
        
        descriptions = {
            "graph_type": "WTA",
            "n": str(self.n),
            "edge_prob": str(self.probability),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }
        
        return MosaicHWMappingInput(
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            assignment=None
        )
