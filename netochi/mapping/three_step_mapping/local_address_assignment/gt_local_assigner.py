from collections import defaultdict

import numpy as np
import numpy.typing as npt

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput


class GroundTruthOrRandomLocalAddressAssigner(LocalAddressAssigner):
    """
    Assigns local addresses using ground truth if available.
    Otherwise, falls back to a random permutation of local indices within each cluster core.
    """

    def assign_addresses(self, mapping_input: MosaicMappingInput, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:
        # 1. Try to use the ground truth if available
        if mapping_input.assignment is not None:
            return np.array(mapping_input.assignment.neuron_idx_pre_assignment, dtype=np.int_)

        # 2. Fallback: Random assignment per core
        num_neurons = mapping_input.graph.num_vertices()
        neuron_local_assignment = np.zeros(num_neurons, dtype=np.int_)

        # Group neuron IDs by their assigned core
        core_to_nodes = defaultdict(list)
        for v in range(num_neurons):
            core_to_nodes[clustering.cluster_assignment[v]].append(v)

        # Assign unique random indices [0, len(nodes)-1] to neurons inside each core
        for core, nodes in core_to_nodes.items():
            if not nodes:
                continue

            # Generate a random shuffle of local indices for the current core size
            random_local_indices = np.random.permutation(len(nodes))

            # Vectorized assignment to the target neurons
            neuron_local_assignment[np.array(nodes)] = random_local_indices

        return neuron_local_assignment

    def get_name(self) -> str:
        return "Gt"