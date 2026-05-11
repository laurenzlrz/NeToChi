from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class SwtaGeneratorConfig:

    def __init__(self,
                 num_clusters: int,
                 neurons_per_cluster: int,
                 inhibitory_ratio: float = 0.2,  # 20% of clusters are inhibitory
                 p_neighbor: float = 0.1,  # Neighboring E-cluster connection
                 p_e_to_i: float = 0.2,  # Excitatory to Inhibitory
                 p_i_to_e: float = 0.8,  # Inhibitory to Excitatory (Blanket inhibition)
                 seed: int | None = None):
        self._num_clusters = num_clusters
        self._neurons_per_cluster = neurons_per_cluster
        self._inhibitory_ratio = inhibitory_ratio
        self._p_neighbor = p_neighbor
        self._p_e_to_i = p_e_to_i
        self._p_i_to_e = p_i_to_e
        self._seed = seed

    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    @property
    def neurons_per_cluster(self) -> int:
        return self._neurons_per_cluster

    @property
    def inhibitory_ratio(self) -> float:
        return self._inhibitory_ratio

    @property
    def p_neighbor(self) -> float:
        return self._p_neighbor

    @property
    def p_e_to_i(self) -> float:
        return self._p_e_to_i

    @property
    def p_i_to_e(self) -> float:
        return self._p_i_to_e

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def num_i_clusters(self) -> int:
        return max(1, int(self._num_clusters * self._inhibitory_ratio)) # Ensures at least 1 inhibitory cluster if ratio > 0

    @property
    def num_e_clusters(self) -> int:
        return self._num_clusters - self.num_i_clusters

    @property
    def total_neurons(self) -> int:
        return self._num_clusters * self._neurons_per_cluster


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

    def _are_clusters_neighbors(self, src_cluster: int, tgt_cluster: int):
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