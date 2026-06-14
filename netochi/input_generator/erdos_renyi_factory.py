import networkx as nx
from pydantic import BaseModel, Field, ConfigDict, validate_call

from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt


class ErdosRenyiFactory(BaseModel, HWBaseInputFactory[MosaicMappingInput]):
    """Factory generating Erdős-Rényi networks for a fixed hardware configuration."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        strict=True
    )

    hw_config: MosaicHardwareConfig
    n: int = Field(gt=0, description="Number of nodes.")
    probability: float = Field(ge=0, le=1, description="Edge creation probability.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    def get_name(self) -> str:
        """Returns a concise name reflecting size and probability."""
        return f"ER_{self.n}n_p{self.probability}"

    @validate_call
    def generate(self) -> MosaicMappingInput:
        """Generate a single MosaicMappingInput with an Erdős-Rényi graph."""
        graph = nx.fast_gnp_random_graph(self.n, self.probability, seed=self.seed, directed=True)
        gt_graph = nx_to_gt(graph)
        
        descriptions = {
            "graph_type": self.get_name(),
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

    def get_id(self) -> str:
        return f"ErdosRenyi_n={self.n}_p={self.probability}_seed={self.seed}"
