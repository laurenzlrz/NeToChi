from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Tuple, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import icontract

from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt


class MosaicNetworkConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    hw_config: MosaicHardwareConfig = Field(..., description="Hardware configuration.")
    probability: float = Field(..., gt=0, le=1)
    seed: int = Field()
    permute: bool = Field(default=True, description="Whether to permute the graph node IDs after generation.")

    def create(self) -> "MosaicNetworkFactory":
        return MosaicNetworkFactory(config=self)


class MosaicNetworkFactory(HWBaseInputFactory[MosaicMappingInput]):
    """Factory generating synthetic networks for a fixed hardware configuration."""

    @icontract.require(lambda config: isinstance(config, MosaicNetworkConfig))
    def __init__(self, config: MosaicNetworkConfig) -> None:
        self.config = config
        self.hw_config = config.hw_config
        self.probability = config.probability
        self.seed = config.seed
        self._rng = np.random.default_rng(self.seed)
        self.permute = config.permute


    def get_id(self) -> str:
        """Returns a concise name reflecting size and probability."""
        return f"Mosaic_{self.hw_config.total_neurons}n_{self.hw_config.nodes_per_router}R_{self.hw_config.router_levels}l_{self.hw_config.neurons_per_core}N_{self.probability}p"

    def generate(self) -> MosaicMappingInput:
        """Generate a single MosaicMappingInput."""
        graph, assignment = self._generate_network()
        gt_graph = nx_to_gt(graph)
        
        descriptions = {
            "graph_type": self.get_id(),
            "edge_prob": str(self.probability),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }

        return MosaicMappingInput(
            id=self.get_id(),
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            assignment=assignment,
            seed=self.seed
        )

    def _sample_slice_assignment(self) -> npt.NDArray[np.int64]:
        """Randomly assign fan-in slices to each target neuron at each router level."""
        total_neurons: int = self.hw_config.total_neurons
        router_levels: int = self.hw_config.router_levels
        assignment: npt.NDArray[np.int64] = np.zeros(
            (total_neurons, router_levels + 1),
            dtype=np.int64
        ).astype(np.int64)

        for distance in range(0, router_levels + 1):
            slices: int = self.hw_config.num_slices_at_distance(distance)
            assignment[:, distance] = self._rng.integers(0, slices, size=total_neurons)

        return assignment

    def _generate_network(self) -> Tuple[nx.DiGraph, MosaicAssignment]:
        """Perform the actual network generation using the Fan-In constraint."""
        slice_assignment = self._sample_slice_assignment()
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(range(self.hw_config.total_neurons))

        total_neurons: int = self.hw_config.total_neurons
        neurons_per_core: int = self.hw_config.neurons_per_core
        total_cores: int = self.hw_config.total_cores

        # --- NEW: Define permutation maps ---
        if self.permute:
            perm = self._rng.permutation(total_neurons)
            inv_perm = np.argsort(perm)
        else:
            perm = np.arange(total_neurons)
            inv_perm = perm

        for target_neuron in range(total_neurons):
            target_core: int = target_neuron // neurons_per_core

            for source_core in range(total_cores):
                distance: int = self.hw_config.core_distance(source_core, target_core)

                chosen_slice: int = int(slice_assignment[target_neuron, distance])
                local_start, local_end = self.hw_config.get_slice_bounds(distance, chosen_slice)

                source_candidates = np.arange(
                    source_core * neurons_per_core + local_start,
                    source_core * neurons_per_core + local_end,
                    dtype=np.int64,
                )

                if source_core == target_core:
                    source_candidates = source_candidates[source_candidates != target_neuron]

                if source_candidates.size == 0:
                    continue

                sampled = self._rng.random(source_candidates.size) < self.probability
                selected_sources = source_candidates[sampled]

                # --- CHANGED: Apply perturbation to external graph node IDs ---
                graph.add_edges_from((int(perm[src]), int(perm[target_neuron])) for src in selected_sources)

        # --- CHANGED: Align assignments with perturbed graph IDs via inv_perm ---
        assignment: MosaicAssignment = MosaicAssignment(
            hw=self.hw_config,
            neuron_core_pre_assignment=inv_perm // neurons_per_core,
            neuron_idx_pre_assignment=inv_perm % neurons_per_core,
            neuron_slice_assignment=slice_assignment[inv_perm]
        )
        return graph, assignment
