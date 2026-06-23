from pydantic import BaseModel, Field, ConfigDict
import networkx as nx
import icontract

from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt


class ErdosRenyiConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    hw_config: MosaicHardwareConfig = Field(..., description="Hardware configuration.")
    n: int = Field(gt=0, description="Number of nodes.")
    probability: float = Field(ge=0, le=1, description="Edge creation probability.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    def create(self) -> "ErdosRenyiFactory":
        return ErdosRenyiFactory(config=self)


class ErdosRenyiFactory(HWBaseInputFactory[MosaicMappingInput]):
    """Factory generating Erdős-Rényi networks for a fixed hardware configuration."""

    @icontract.require(lambda config: isinstance(config, ErdosRenyiConfig))
    def __init__(self, config: ErdosRenyiConfig) -> None:
        self.config = config
        self.hw_config = config.hw_config
        self.n = config.n
        self.probability = config.probability
        self.seed = config.seed


    def get_name(self) -> str:
        """Returns a concise name reflecting size and probability."""
        return f"ER_{self.n}n_p{self.probability}"

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
