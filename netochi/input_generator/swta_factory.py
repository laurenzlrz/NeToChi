from dataclasses import dataclass
from typing import Optional, Any
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt

@dataclass
class SwtaGeneratorConfig:
    num_clusters: int
    neurons_per_cluster: int
    inhibitory_ratio: float = 0.2  # 20% of clusters are inhibitory
    p_neighbor: float = 0.1  # Neighboring E-cluster connection
    p_e_to_i: float = 0.2  # Excitatory to Inhibitory
    p_i_to_e: float = 0.8  # Inhibitory to Excitatory (Blanket inhibition)
    seed: Optional[int] = None

    @property
    def num_i_clusters(self) -> int:
        return max(1, int(self.num_clusters * self.inhibitory_ratio)) # Ensures at least 1 inhibitory cluster if ratio > 0

    @property
    def num_e_clusters(self) -> int:
        return self.num_clusters - self.num_i_clusters

    @property
    def total_neurons(self) -> int:
        return self.num_clusters * self.neurons_per_cluster


@dataclass(frozen=True)
class SwtaNetworkResult:
    graph: nx.DiGraph
    config: SwtaGeneratorConfig



class SwtaNetwork:
    """
    represents sWTA networks described in this paper (Fig.10, clustered WTA): https://iopscience.iop.org/article/10.1088/2634-4386/ace64c/meta
    """

    def __init__(self, config: SwtaGeneratorConfig):
        self._rng = np.random.default_rng(config.seed) # random number generator
        self._graph: nx.DiGraph | None = None # actual network, generated with generate()
        self._config: SwtaGeneratorConfig = config # blueprint for network generation

    def _get_cluster_id(self, neuron_idx: int) -> int:
        return neuron_idx // self._config.neurons_per_cluster

    def _are_clusters_neighbors(self, src_cluster: int, tgt_cluster: int) -> bool:
        return abs(src_cluster - tgt_cluster) == 1

    def _is_inhibitory(self, cluster_id: int) -> bool:
        return cluster_id >= self._config.num_e_clusters # last few clusters are Inhibitory

    def _compute_connection_probability(self, src_cluster: int, tgt_cluster: int, tgt_is_i: bool, src_is_i: bool) -> float:
        prob = 0.0
        if src_cluster == tgt_cluster:
            # 1. Intra-cluster: All-to-all recurrent connections
            prob = 1.0
        elif not src_is_i and not tgt_is_i:
            # 2. Connections between neighboring Excitatory clusters
            if self._are_clusters_neighbors(src_cluster, tgt_cluster):
                prob = self._config.p_neighbor
        elif not src_is_i and tgt_is_i:
            # 3. Excitatory to Inhibitory (Sensing global activity)
            prob = self._config.p_e_to_i
        elif src_is_i and not tgt_is_i:
            # 4. Inhibitory to Excitatory (The "Blanket" of inhibition)
            prob = self._config.p_i_to_e
        return prob

    def generate(self) -> nx.DiGraph:
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(range(self._config.total_neurons))

        for src in range(self._config.total_neurons):
            src_cluster = self._get_cluster_id(src)
            src_is_i = self._is_inhibitory(src_cluster)

            for tgt in range(self._config.total_neurons):
                if src == tgt: continue  # No self-loops
                tgt_cluster = self._get_cluster_id(tgt)
                tgt_is_i = self._is_inhibitory(tgt_cluster)
                prob = self._compute_connection_probability(src_cluster, tgt_cluster, tgt_is_i, src_is_i)

                if prob > 0 and (prob == 1.0 or self._rng.random() < prob):
                    self._graph.add_edge(src, tgt)

        return self._graph


class SwtaFactory(BaseModel, HWBaseInputFactory[MosaicMappingInput[SwtaNetworkResult]]):
    """
    Factory generating soft Winner-Takes-All (sWTA) networks.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    hw_config: MosaicHardwareConfig
    num_clusters: int = Field(..., gt=0)
    neurons_per_cluster: int = Field(..., gt=0)
    inhibitory_ratio: float = 0.2
    p_neighbor: float = 0.1
    p_e_to_i: float = 0.2
    p_i_to_e: float = 0.8
    seed: int = 42

    def generate(self) -> MosaicMappingInput[SwtaNetworkResult]:
        """Generate a single MosaicMappingInput with an sWTA graph."""
        config = SwtaGeneratorConfig(
            num_clusters=self.num_clusters,
            neurons_per_cluster=self.neurons_per_cluster,
            inhibitory_ratio=self.inhibitory_ratio,
            p_neighbor=self.p_neighbor,
            p_e_to_i=self.p_e_to_i,
            p_i_to_e=self.p_i_to_e,
            seed=self.seed
        )
        swta = SwtaNetwork(config)
        nx_graph = swta.generate()
        gt_graph = nx_to_gt(nx_graph)
        
        descriptions = {
            "graph_type": "sWTA",
            "num_clusters": str(self.num_clusters),
            "neurons_per_cluster": str(self.neurons_per_cluster),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }
        
        return MosaicMappingInput(
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            payload=SwtaNetworkResult(graph=nx_graph, config=config),
            pre_assignment=None
        )