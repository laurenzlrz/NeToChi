from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class RandomGeneratorConfig:

    def __init__(self,
                 nodes_per_router: int,
                 neurons_per_core: int,
                 router_levels: int,
                 edge_probability: float,
                 slice_factor: int = 2,
                 seed: int | None = None):
        if nodes_per_router < 1:
            raise ValueError("nodes_per_router must be >= 1")
        if neurons_per_core < 1:
            raise ValueError("neurons_per_core must be >= 1")
        if router_levels < 1:
            raise ValueError("router_levels must be >= 1")
        if not (0.0 <= edge_probability <= 1.0):
            raise ValueError("edge_probability must be in [0.0, 1.0]")

        self._nodes_per_router: int = nodes_per_router
        self._neurons_per_core: int = neurons_per_core
        self._router_levels: int = router_levels
        self._slice_factor: int = slice_factor   # Each level slices the core's neurons into this many parts.
        self._edge_probability: float = edge_probability
        self._seed: int | None = seed

    @property
    def nodes_per_router(self) -> int:
        return self._nodes_per_router

    @property
    def neurons_per_core(self) -> int:
        return self._neurons_per_core

    @property
    def router_levels(self) -> int:
        return self._router_levels

    @property
    def edge_probability(self) -> float:
        return self._edge_probability

    @property
    def total_cores(self) -> int:
        return self.nodes_per_router ** self.router_levels

    @property
    def total_neurons(self) -> int:
        return self.total_cores * self.neurons_per_core

    @property
    def slice_factor(self) -> int:
        return self._slice_factor

    @property
    def seed(self) -> int | None:
        return self._seed


@dataclass(frozen=True)
class RandomNetworkResult:
    graph: nx.DiGraph
    config: RandomGeneratorConfig
    assignment: np.ndarray

@dataclass
class NeuronClass:

    def __init__(self, idx: int, core: int, slices: list[int]):
        self._idx = idx
        self._core = core
        self._slices = slices

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def core(self) -> int:
        return self._core

    @property
    def slices(self) -> list[int]:
        return self._slices


class RandomNetwork:

    def __init__(self, config: RandomGeneratorConfig):
        self._rng = np.random.default_rng(config.seed)
        #TODO: implement NeuronClass to save more information
        self._graph: nx.DiGraph | None = None
        self._config: RandomGeneratorConfig = config
        self._assignment: np.ndarray | None = None

    def _sample_assignment(self) -> None:
        self._assignment = np.zeros((self._config.total_neurons, self._config.router_levels + 1), dtype=np.int64)

        for distance in range(0, self._config.router_levels + 1):
            slices = min(2 ** distance, self._config.neurons_per_core)
            self._assignment[:, distance] = self._rng.integers(0, slices, size=self._config.total_neurons)


    def _core_distance(self, core_a: int, core_b: int) -> int:
        if core_a == core_b:
            return 0

        base = self._config.nodes_per_router
        levels = self._config.router_levels

        # Hierarchical distance: highest differing router level + 1.
        # levels - 1, because they are the same on the highest level, so no need to check
        for level in range(levels - 1, -1, -1):
            # Level 0: --> base ** level = 1, digit is core itself.
            # Level 1: --> base ** level = base
            digit_a = (core_a // (base ** level)) % base
            digit_b = (core_b // (base ** level)) % base
            if digit_a != digit_b:
                return level + 1

        return 0

    def _slice_bounds(self, slices: int, slice_idx: int) -> tuple[int, int]:
        slices = min(slices, self._config.neurons_per_core) # Cannot have more slices than neurons per core
        start = (slice_idx * self._config.neurons_per_core) // slices
        end = ((slice_idx + 1) * self._config.neurons_per_core) // slices
        return start, end

    def generate(self) -> None:
        self._sample_assignment()
        self._graph = nx.DiGraph()
        #Could be improved in the future with more specific graph type
        self._graph.add_nodes_from(range(self._config.total_neurons))

        for target_neuron in range(self._config.total_neurons):
            target_core = target_neuron // self._config.total_cores

            for source_core in range(self._config.total_cores):
                distance = self._core_distance(source_core, target_core)

                chosen_slice = int(self._assignment[target_neuron, distance])
                local_start, local_end = self._slice_bounds(self._config.slice_factor ** distance, chosen_slice)

                if local_end <= local_start:
                    continue

                source_candidates = np.arange(
                    source_core * self._config.neurons_per_core + local_start,
                    source_core * self._config.neurons_per_core + local_end,
                    dtype=np.int64,
                )

                # No self-loop; intra-core is otherwise fully connected at distance 0.
                if source_core == target_core:
                    source_candidates = source_candidates[source_candidates != target_neuron]

                if source_candidates.size == 0:
                    continue

                #Mask Out-Simulation with random sampling
                sampled = self._rng.random(source_candidates.size) < self._config.edge_probability # Binary Array
                selected_sources = source_candidates[sampled] # Sampled sources for this target neuron with boolean mask: Contains for one core the sources
                self._graph.add_edges_from((int(src), target_neuron) for src in selected_sources)

