from typing import Any, Optional
import networkx as nx
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from netochi.input_generator.interfaces import MosaicHWMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt

class ErdosRenyiFactory(BaseModel, HWBaseInputFactory[MosaicHWMappingInput[Any]]):
    """Factory generating Erdős-Rényi networks for a fixed hardware configuration."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    hw_config: MosaicHardwareConfig
    n: int = Field(..., gt=0, description="Number of nodes.")
    probability: float = Field(..., ge=0, le=1, description="Edge creation probability.")
    seed: int = 42
    
    _graph: Optional[nx.DiGraph] = PrivateAttr(default=None)

    def generate(self) -> MosaicHWMappingInput[Any]:
        """Generate a single MosaicMappingInput with an Erdős-Rényi graph."""
        graph = nx.fast_gnp_random_graph(self.n, self.probability, seed=self.seed, directed=True)
        object.__setattr__(self, '_graph', graph)
        
        gt_graph = nx_to_gt(graph)
        
        descriptions = {
            "graph_type": "ErdosRenyi",
            "n": str(self.n),
            "edge_prob": str(self.probability),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }
        
        return MosaicHWMappingInput(
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            payload=None,
            pre_assignment=None  # ER graphs don't have a default hardware mapping
        )

    def get_id(self):
        return f"ErdosRenyi_n={self.n}_p={self.probability}_seed={self.seed}"

