"""
sWTA generator, version 2: cluster sizes are chosen so that the network fits the hardware exactly.

- One excitatory (E) cluster per core, arranged as a chain E_0 - E_1 - ... - E_{C-1}.
- One global inhibitory (I) cluster of size C * q, where q = inhibitory neurons per core.
- Every E cluster has n - q neurons (n = neurons per core), so an E cluster together with its
  share of q inhibitory neurons fills exactly one core.

Connectivity (same rules as SwtaFactory):
- within every cluster (E or I): all-to-all, no self-loops
- E_k -> E_{k +- 1}: p_neighbor
- E -> I: p_e_to_i
- I -> E: p_i_to_e

Node order: E_0, E_1, ..., E_{C-1} (each n - q neurons), then the I cluster.
"""
from typing import Optional

import graph_tool.all as gt
import icontract
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netochi.definitions.exceptions import InvalidConfigError
from netochi.input_generator.interfaces import HWBaseInputFactory, MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


class SwtaV2GeneratorConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    hw_config: MosaicHardwareConfig = Field(..., description="Hardware configuration.")
    i_neurons_per_core: Optional[int] = Field(
        default=None, gt=0,
        description="Inhibitory neurons per core (q). If None: neurons_per_core // total_cores (at least 1), "
                    "which is the largest q for which all I neurons fit into slice 0 at the maximal distance.")
    p_neighbor: float = Field(default=0.1, ge=0.0, le=1.0, description="Neighboring E-cluster connection probability")
    p_e_to_i: float = Field(default=0.2, ge=0.0, le=1.0, description="Excitatory to Inhibitory probability")
    p_i_to_e: float = Field(default=0.8, ge=0.0, le=1.0, description="Inhibitory to Excitatory blanket probability")
    seed: Optional[int] = Field(description="Random seed")

    @model_validator(mode="after")
    def validate_sizes(self) -> "SwtaV2GeneratorConfig":
        if self.i_per_core >= self.hw_config.neurons_per_core:
            raise InvalidConfigError("i_neurons_per_core must be smaller than neurons_per_core, "
                                     "otherwise the excitatory clusters are empty.")
        return self

    @property
    def i_per_core(self) -> int:
        if self.i_neurons_per_core is not None:
            return self.i_neurons_per_core
        return max(1, self.hw_config.neurons_per_core // self.hw_config.total_cores)

    @property
    def num_e_clusters(self) -> int:
        return self.hw_config.total_cores

    @property
    def e_cluster_size(self) -> int:
        return self.hw_config.neurons_per_core - self.i_per_core

    @property
    def i_cluster_size(self) -> int:
        return self.hw_config.total_cores * self.i_per_core

    @property
    def total_neurons(self) -> int:
        return self.num_e_clusters * self.e_cluster_size + self.i_cluster_size

    def create(self) -> "SwtaV2Factory":
        return SwtaV2Factory(config=self)


class SwtaV2Factory(HWBaseInputFactory[MosaicMappingInput]):
    """Factory generating sWTA networks whose clusters fit the hardware (see module docstring)."""

    @icontract.require(lambda config: isinstance(config, SwtaV2GeneratorConfig))
    def __init__(self, config: SwtaV2GeneratorConfig) -> None:
        self.config = config
        self.hw_config = config.hw_config
        self._rng = np.random.default_rng(config.seed)

    def get_id(self) -> str:
        c = self.config
        return f"sWTAv2_{c.total_neurons}n_{c.num_e_clusters}E{c.e_cluster_size}_I{c.i_cluster_size}"

    def cluster_labels(self) -> npt.NDArray[np.int64]:
        """Cluster per neuron: 0..C-1 for the E chain, C for the inhibitory cluster."""
        c = self.config
        e_labels = np.repeat(np.arange(c.num_e_clusters, dtype=np.int64), c.e_cluster_size)
        i_labels = np.full(c.i_cluster_size, c.num_e_clusters, dtype=np.int64)
        return np.concatenate([e_labels, i_labels])

    def _probability_matrix(self, labels: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        c = self.config
        num_clusters = c.num_e_clusters + 1
        inh = c.num_e_clusters
        block = np.zeros((num_clusters, num_clusters))
        np.fill_diagonal(block, 1.0)
        for k in range(c.num_e_clusters - 1):
            block[k, k + 1] = block[k + 1, k] = c.p_neighbor
        block[:inh, inh] = c.p_e_to_i
        block[inh, :inh] = c.p_i_to_e
        prob = block[labels[:, None], labels[None, :]]
        np.fill_diagonal(prob, 0.0)
        return prob

    def generate_network(self) -> gt.Graph:
        labels = self.cluster_labels()
        prob = self._probability_matrix(labels)
        src, tgt = np.nonzero(self._rng.random(prob.shape) < prob)
        graph = gt.Graph(directed=True)
        graph.add_vertex(labels.size)
        graph.add_edge_list(np.column_stack([src, tgt]))
        return graph

    def generate(self) -> MosaicMappingInput:
        graph = self.generate_network()
        c = self.config
        descriptions = {
            "graph_type": self.get_id(),
            "num_e_clusters": str(c.num_e_clusters),
            "e_cluster_size": str(c.e_cluster_size),
            "num_i_clusters": "1",
            "i_cluster_size": str(c.i_cluster_size),
            "i_neurons_per_core": str(c.i_per_core),
            "nodes": str(graph.num_vertices()),
            "edges": str(graph.num_edges()),
        }
        return MosaicMappingInput(
            id=self.get_id(),
            graph=graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            assignment=None,
            seed=c.seed,
        )
