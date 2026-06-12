from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict
from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.definitions.exceptions import HardwareConstraintError
from netochi.definitions.constants import HARDWARE_CAPACITY_EXCEEDED


class GreedyMapper(BaseModel, BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper implementing an optimized dynamic-frontier greedy heuristic
    to maximize topological clustering on hardware cores.

    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        """
        Execute the dynamic-frontier greedy mapping algorithm.
        """
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        num_neurons = graph.num_vertices()

        # 1. Initialize result arrays
        core_assignments = np.full(num_neurons, -1, dtype=np.int64)
        local_assignments = np.full(num_neurons, -1, dtype=np.int64)
        slice_assignments = np.zeros((num_neurons, hw.router_levels + 1), dtype=np.int64)

        # Track current core capacity
        core_occupancy = np.zeros(hw.total_cores, dtype=np.int64)

        # Base metrics for tie-breaking
        in_degrees = graph.get_in_degrees(graph.get_vertices())
        out_degrees = graph.get_out_degrees(graph.get_vertices())
        total_degrees = in_degrees + out_degrees

        # Dynamic tracking state
        unplaced_mask = np.ones(num_neurons, dtype=bool)
        # pull_scores[i] = how many neighbors of i have already been placed
        pull_scores = np.zeros(num_neurons, dtype=np.int64)

        # 2. Phase I: Dynamic Core and Local Index Assignment
        for _ in range(num_neurons):
            # Get all unplaced node indices
            candidates = np.where(unplaced_mask)[0]

            # Find candidate(s) with maximum pull from already placed clusters
            candidate_pulls = pull_scores[candidates]
            max_pull = np.max(candidate_pulls)
            best_candidates = candidates[candidate_pulls == max_pull]

            # Tie-breaker: pick the one with the highest global degree
            if len(best_candidates) > 1:
                best_node = best_candidates[np.argmax(total_degrees[best_candidates])]
            else:
                best_node = best_candidates[0]

            # Mark as placed
            unplaced_mask[best_node] = False

            # Score cores based on where neighbors are already placed
            core_scores = np.zeros(hw.total_cores)
            for neighbor in graph.get_all_neighbors(best_node):
                neighbor_core = core_assignments[int(neighbor)]
                if neighbor_core != -1:
                    core_scores[neighbor_core] += 1

            # Find the best available core (highest neighbor count, has space)
            best_cores_sorted = np.argsort(core_scores)[::-1]
            assigned = False
            for core_idx in best_cores_sorted:
                if core_occupancy[core_idx] < hw.neurons_per_core:
                    core_assignments[best_node] = core_idx
                    local_assignments[best_node] = core_occupancy[core_idx]
                    core_occupancy[core_idx] += 1
                    assigned = True
                    break

            if not assigned:
                raise HardwareConstraintError(
                    HARDWARE_CAPACITY_EXCEEDED.format(
                        node=best_node,
                        total=num_neurons,
                        capacity=hw.total_neurons
                    )
                )

            # Update pull scores for the unplaced neighbors of our newly placed node
            for neighbor in graph.get_all_neighbors(best_node):
                neighbor_idx = int(neighbor)
                if unplaced_mask[neighbor_idx]:
                    pull_scores[neighbor_idx] += 1

        # 3. Phase II: Greedy Slice Selection (Fan-In Optimization)
        for target_node in range(num_neurons):
            target_core = core_assignments[target_node]

            for d in range(1, hw.router_levels + 1):
                num_slices = hw.num_slices_at_distance(d)
                best_slice = 0
                max_valid_edges = -1

                for s_idx in range(num_slices):
                    valid_count = 0
                    start, end = hw.get_slice_bounds(d, s_idx)

                    for source_node in graph.get_in_neighbors(target_node):
                        source_core = core_assignments[int(source_node)]
                        if hw.core_distance(target_core, source_core) == d:
                            source_local = local_assignments[int(source_node)]
                            if start <= source_local < end:
                                valid_count += 1

                    if valid_count > max_valid_edges:
                        max_valid_edges = valid_count
                        best_slice = s_idx

                slice_assignments[target_node, d] = best_slice

        return MosaicNetworkMappingState(
            _mapping_input=mapping_input,
            assignment=MosaicAssignment(
                hw=hw,
                neuron_core_pre_assignment=core_assignments,
                neuron_idx_pre_assignment=local_assignments,
                neuron_slice_assignment=slice_assignments.as_type(np.int64)
            )
        )