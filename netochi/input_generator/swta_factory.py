from typing import Optional
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr
from pydantic.dataclasses import dataclass as pydantic_dataclass

from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt


@pydantic_dataclass(config=ConfigDict(strict=True))
class SwtaGeneratorConfig:
    num_clusters: int = Field(..., gt=0, description="Total number of clusters")
    neurons_per_cluster: int = Field(..., gt=0, description="Neurons within each cluster")
    inhibitory_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="20% of clusters are inhibitory")
    p_neighbor: float = Field(default=0.1, ge=0.0, le=1.0, description="Neighboring E-cluster connection probability")
    p_e_to_i: float = Field(default=0.2, ge=0.0, le=1.0, description="Excitatory to Inhibitory probability")
    p_i_to_e: float = Field(default=0.8, ge=0.0, le=1.0, description="Inhibitory to Excitatory blanket probability")
    seed: Optional[int] = Field(default=None, description="Random seed")

    @property
    def num_i_clusters(self) -> int:
        return max(1, int(self.num_clusters * self.inhibitory_ratio))

    @property
    def num_e_clusters(self) -> int:
        return self.num_clusters - self.num_i_clusters

    @property
    def total_neurons(self) -> int:
        return self.num_clusters * self.neurons_per_cluster




class SwtaFactory[PAYLOAD](BaseModel, HWBaseInputFactory[MosaicMappingInput[PAYLOAD]]):
    """
    Factory generating soft Winner-Takes-All (sWTA) networks.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        strict=True
    )

    config: SwtaGeneratorConfig = Field(..., description="Blueprint for network generation")
    hw_config: MosaicHardwareConfig = Field(..., description="Hardware configuration")
    _rng: np.random.Generator = PrivateAttr()

    def get_name(self) -> str:
        """Returns a name reflecting size and configuration."""
        total_n = self.config.num_clusters * self.config.neurons_per_cluster
        return f"sWTA_{total_n}n_c{self.config.num_clusters}_npc{self.config.neurons_per_cluster}"

    @model_validator(mode="after")
    def initalize_rng(self) -> 'SwtaFactory[PAYLOAD]':
        """Initialize the private random number generator after Pydantic validates inputs."""
        self.__setattr__("_rng", np.random.default_rng(self.config.seed))
        return self

    def _get_cluster_id(self, neuron_idx: int) -> int:
        return neuron_idx // self.config.neurons_per_cluster

    def _are_clusters_neighbors(self, src_cluster: int, tgt_cluster: int) -> bool:
        return abs(src_cluster - tgt_cluster) == 1

    def _is_inhibitory(self, cluster_id: int) -> bool:
        return cluster_id >= self.config.num_e_clusters

    def _compute_connection_probability(self, src_cluster: int, tgt_cluster: int, tgt_is_i: bool,
                                        src_is_i: bool) -> float:
        prob = 0.0
        if src_cluster == tgt_cluster:
            prob = 1.0
        elif not src_is_i and not tgt_is_i:
            if self._are_clusters_neighbors(src_cluster, tgt_cluster):
                prob = self.config.p_neighbor
        elif not src_is_i and tgt_is_i:
            prob = self.config.p_e_to_i
        elif src_is_i and not tgt_is_i:
            prob = self.config.p_i_to_e
        return prob

    def generate_network(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.config.total_neurons))

        for src in range(self.config.total_neurons):
            src_cluster = self._get_cluster_id(src)
            src_is_i = self._is_inhibitory(src_cluster)

            for tgt in range(self.config.total_neurons):
                if src == tgt:
                    continue
                tgt_cluster = self._get_cluster_id(tgt)
                tgt_is_i = self._is_inhibitory(tgt_cluster)
                prob = self._compute_connection_probability(src_cluster, tgt_cluster, tgt_is_i, src_is_i)

                if prob > 0 and (prob == 1.0 or self._rng.random() < prob):
                    graph.add_edge(src, tgt)

        return graph

    def generate(self) -> MosaicMappingInput[PAYLOAD]:
        """Generate a single MosaicMappingInput with an sWTA graph."""

        nx_graph = self.generate_network()
        gt_graph = nx_to_gt(nx_graph)

        descriptions = {
            "graph_type": self.get_name(),
            "num_clusters": str(self.config.num_clusters),
            "neurons_per_cluster": str(self.config.neurons_per_cluster),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }

        return MosaicMappingInput(
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            payload=None,
            assignment=None
        )