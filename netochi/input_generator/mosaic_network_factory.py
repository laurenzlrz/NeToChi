from typing import Any, Tuple, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict, validate_call, model_validator, PrivateAttr

from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory, MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt


class MosaicNetworkFactory(BaseModel, HWBaseInputFactory[MosaicMappingInput]):
    """Factory generating synthetic networks for a fixed hardware configuration."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        strict=True
    )

    hw_config: MosaicHardwareConfig
    probability: float = Field(..., gt=0, le=1)
    seed: int = 42
    _rng: np.random.Generator = PrivateAttr()

    @model_validator(mode="after")
    def initialize_rng(self) -> "MosaicNetworkFactory":
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))
        return self

    def get_name(self) -> str:
        """Returns a concise name reflecting size and probability."""
        return f"FeasibleMosaicNetwork_{self.hw_config.total_neurons}n_p{self.probability}"

    @validate_call
    def generate(self) -> MosaicMappingInput:
        """Generate a single MosaicMappingInput."""
        graph, assignment = self._generate_network()
        gt_graph = nx_to_gt(graph)
        
        descriptions = {
            "graph_type": self.get_name(),
            "edge_prob": str(self.probability),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }


        return MosaicMappingInput(
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            assignment=assignment # fully defines the ground truth (because core and local address assignment only depend on neuron id)
        )

    @validate_call
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

        for target_neuron in range(total_neurons):
            target_core: int = target_neuron // neurons_per_core

            for source_core in range(total_cores):
                distance: int = self.hw_config.core_distance(source_core, target_core)

                chosen_slice: int = int(slice_assignment[target_neuron, distance])
                local_start, local_end = self.hw_config.get_slice_bounds(distance, chosen_slice)

                #neuron idxs
                source_candidates = np.arange(
                    source_core * neurons_per_core + local_start,
                    source_core * neurons_per_core + local_end,
                    dtype=np.int64,
                )

                # No self-loop; intra-core is otherwise fully connected at distance 0.
                if source_core == target_core:
                    source_candidates = source_candidates[source_candidates != target_neuron]

                if source_candidates.size == 0:
                    continue

                # Sample edges based on probability
                sampled = self._rng.random(source_candidates.size) < self.probability
                selected_sources = source_candidates[sampled]
                graph.add_edges_from((int(src), target_neuron) for src in selected_sources)

        assignment: MosaicAssignment = MosaicAssignment(
            hw=self.hw_config,
            neuron_core_pre_assignment=np.arange(total_neurons) // neurons_per_core,
            neuron_idx_pre_assignment=np.arange(total_neurons) % neurons_per_core,
            neuron_slice_assignment=slice_assignment
        )
        return graph, assignment