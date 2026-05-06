from typing import Dict, Any, Optional
import graph_tool.all as gt
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from netochi.input_generator.interfaces import BaseInputFactory, MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.utils import nx_to_gt

class MosaicNetworkFactory(BaseModel, BaseInputFactory):
    """Factory generating synthetic networks for a fixed hardware configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hw_config: MosaicHardwareConfig
    probability: float = Field(..., gt=0, le=1)
    seed: int = 42
    
    _assignment: Optional[np.ndarray] = PrivateAttr(default=None)
    _graph: Optional[nx.DiGraph] = PrivateAttr(default=None)
    _rng: Any = PrivateAttr(default=None)

    def generate(self) -> MosaicMappingInput[None]:
        """Generate a single MosaicMappingInput."""
        self._rng = np.random.default_rng(self.seed)
        self._generate_network()
        
        # Convert networkx to graph-tool
        gt_graph = nx_to_gt(self._graph)
        
        descriptions = {
            "graph_type": "MosaicRandomNetwork",
            "edge_prob": str(self.probability),
            "nodes": str(gt_graph.num_vertices()),
            "edges": str(gt_graph.num_edges())
        }
        
        return MosaicMappingInput(
            graph=gt_graph,
            descriptions=descriptions,
            hw_config=self.hw_config,
            payload=None,
            pre_assignment=self._assignment
        )

    def _sample_slice_assignment(self) -> None:
        """Randomly assign fan-in slices to each target neuron at each router level."""
        total_neurons = self.hw_config.total_neurons
        router_levels = self.hw_config.router_levels
        self._assignment = np.zeros((total_neurons, router_levels + 1), dtype=np.int64)

        for distance in range(0, router_levels + 1):
            slices = self.hw_config.num_slices_at_distance(distance)
            self._assignment[:, distance] = self._rng.integers(0, slices, size=total_neurons)

    def _generate_network(self) -> None:
        """Perform the actual network generation using the Fan-In constraint."""
        self._sample_slice_assignment()
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(range(self.hw_config.total_neurons))

        total_neurons = self.hw_config.total_neurons
        neurons_per_core = self.hw_config.neurons_per_core
        total_cores = self.hw_config.total_cores

        for target_neuron in range(total_neurons):
            target_core = target_neuron // neurons_per_core

            for source_core in range(total_cores):
                distance = self.hw_config.core_distance(source_core, target_core)

                # Get the slice this target neuron is listening to for this source core
                chosen_slice = int(self._assignment[target_neuron, distance])
                local_start, local_end = self.hw_config.get_slice_bounds(distance, chosen_slice)

                if local_end <= local_start:
                    continue

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
                self._graph.add_edges_from((int(src), target_neuron) for src in selected_sources)
