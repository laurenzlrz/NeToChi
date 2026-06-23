from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import networkx as nx
import numpy as np
import icontract

from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt


class WTAConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    hw_config: MosaicHardwareConfig = Field(..., description="Hardware configuration.")
    n: int = Field(..., ge=2, description="Total number of neurons (nodes).")
    probability: float = Field(..., ge=0, le=1, description="Connectivity probability.")
    seed: int = Field(default=42)

    def create(self) -> "WTAFactory":
        return WTAFactory(config=self)


class WTAFactory(HWBaseInputFactory[MosaicMappingInput]):
    """
    Factory generating Winner-Takes-All (WTA) networks using a hub-and-spoke skeleton.
    - Excitatory Pool: n-1 nodes.
    - Inhibitory Hub: 1 node (the last node).
    """

    @icontract.require(lambda config: isinstance(config, WTAConfig))
    def __init__(self, config: WTAConfig) -> None:
        self.config = config
        self.hw_config = config.hw_config
        self.n = config.n
        self.probability = config.probability
        self.seed = config.seed
        self._graph: Optional[nx.DiGraph] = None


    def generate(self) -> MosaicMappingInput:
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
        self._graph = graph
        
        gt_graph = nx_to_gt(graph)
        
        descriptions = {
            "graph_type": "WTA",
            "n": str(self.n),
            "edge_prob": str(self.probability),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }
        
        return MosaicMappingInput(
            id=self.get_id(),
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            assignment=None
        )

    def get_id(self):
        return f"WTA_n={self.n}_p={self.probability}_seed={self.seed}"

