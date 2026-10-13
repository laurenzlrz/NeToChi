from typing import Tuple

import networkx as nx
import numpy as np
from pydantic import Field

from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment
from netochi.input_generator.utils import nx_to_gt
# adapt the import path to wherever your base generator lives
from netochi.input_generator.mosaic_network_factory import (
    MosaicNetworkConfig,
    MosaicNetworkFactory,
)


class NoisyMosaicNetworkConfig(MosaicNetworkConfig):
    epsilon: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of sampling each hardware-forbidden connection.",
    )

    def create(self) -> "NoisyMosaicNetworkFactory":
        return NoisyMosaicNetworkFactory(config=self)


class NoisyMosaicNetworkFactory(MosaicNetworkFactory):
    """
    Planted Mosaic network (valid edges sampled with `probability`) plus
    noise edges that the hardware cannot realize (sampled with `epsilon`).

    The planted assignment is stored as before. It violates exactly the noise
    edges, so its loss (= number of noise edges) is an upper bound on the optimum.
    """

    def __init__(self, config: NoisyMosaicNetworkConfig) -> None:
        super().__init__(config)
        self.epsilon = config.epsilon
        # Separate RNG for noise: for a fixed seed, the planted part of the graph
        # (slices, permutation, valid edges) is identical for every epsilon.
        self._noise_rng = np.random.default_rng([config.seed, 1])
        self.num_noise_edges = 0
        self.num_valid_edges = 0

    def get_id(self) -> str:
        return f"Noisy_{super().get_id()}_{self.epsilon}eps"

    def generate(self) -> MosaicMappingInput:
        graph, assignment = self._generate_network()
        gt_graph = nx_to_gt(graph)

        descriptions = {
            "graph_type": self.get_id(),
            "edge_prob": str(self.probability),
            "epsilon": str(self.epsilon),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges()),
            "valid_edges": str(self.num_valid_edges),
            "noise_edges": str(self.num_noise_edges),
        }

        return MosaicMappingInput(
            id=self.get_id(),
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            assignment=assignment,
            seed=self.seed,
        )

    def _generate_network(self) -> Tuple[nx.DiGraph, MosaicAssignment]:
        slice_assignment = self._sample_slice_assignment()

        total_neurons = self.hw_config.total_neurons
        neurons_per_core = self.hw_config.neurons_per_core
        total_cores = self.hw_config.total_cores

        if self.permute:
            perm = self._rng.permutation(total_neurons)
            inv_perm = np.argsort(perm)
        else:
            perm = np.arange(total_neurons)
            inv_perm = perm

        valid_edges = []   # (src, tgt) in hardware ids
        noise_edges = []

        for target_neuron in range(total_neurons):
            target_core = target_neuron // neurons_per_core

            for source_core in range(total_cores):
                distance = self.hw_config.core_distance(source_core, target_core)
                chosen_slice = int(slice_assignment[target_neuron, distance])
                local_start, local_end = self.hw_config.get_slice_bounds(distance, chosen_slice)
                base = source_core * neurons_per_core

                # ---- valid edges (inside the chosen slice), same as before ----
                allowed = np.arange(base + local_start, base + local_end, dtype=np.int64)
                if source_core == target_core:
                    allowed = allowed[allowed != target_neuron]
                if allowed.size > 0:
                    keep = self._rng.random(allowed.size) < self.probability
                    valid_edges.extend((int(s), target_neuron) for s in allowed[keep])

                # ---- noise edges (outside the chosen slice) ----
                if self.epsilon > 0.0:
                    forbidden = np.concatenate([
                        np.arange(base, base + local_start, dtype=np.int64),
                        np.arange(base + local_end, base + neurons_per_core, dtype=np.int64),
                    ])
                    if source_core == target_core:
                        forbidden = forbidden[forbidden != target_neuron]
                    if forbidden.size > 0:
                        keep = self._noise_rng.random(forbidden.size) < self.epsilon
                        noise_edges.extend((int(s), target_neuron) for s in forbidden[keep])

        self.num_valid_edges = len(valid_edges)
        self.num_noise_edges = len(noise_edges)

        graph = nx.DiGraph()
        graph.add_nodes_from(range(total_neurons))
        graph.add_edges_from((int(perm[s]), int(perm[t])) for s, t in valid_edges)
        graph.add_edges_from((int(perm[s]), int(perm[t])) for s, t in noise_edges)

        assignment = MosaicAssignment(
            hw=self.hw_config,
            neuron_core_pre_assignment=inv_perm // neurons_per_core,
            neuron_idx_pre_assignment=inv_perm % neurons_per_core,
            neuron_slice_assignment=slice_assignment[inv_perm],
        )
        return graph, assignment