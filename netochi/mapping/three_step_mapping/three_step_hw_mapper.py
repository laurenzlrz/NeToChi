
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, SliceAssigner, \
    ClusterAndHwOutput, ClustererFixedHw

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, MosaicAssignment
import numpy as np


class ThreeStepHwMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Given mapping input + hardware, infers mapping
    """

    def __init__(self, clusterer: ClustererFixedHw, address_assigner: LocalAddressAssigner, slice_assigner: SliceAssigner):
        self._clusterer = clusterer
        self._address_assigner = address_assigner
        self._slice_assigner = slice_assigner


    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        """
        the mapping runs in three stages:
            1. clustering
            2. Local address assignment: assigns each neuron a local address within its core
            3. Slice assignment: assigns each (neuron, distance) a slice it listens to
        """
        graph = mapping_input.graph

        # --- 1. Clustering ---
        clustering: ClusterAndHwOutput = self._clusterer.cluster(input_data=mapping_input)

        # --- 2. Local address assignment ---
        neuron_local_assignment = self._address_assigner.assign_addresses(graph=graph, clustering=clustering)

        # --- 3. Slice assignment ---
        neuron_slice_assignment = self._slice_assigner.assign_slices(clustering=clustering, graph=graph, local_assignment=neuron_local_assignment)

        # --- 4. Create Mapping State ---
        assignment = MosaicAssignment(
            hw=mapping_input.hw_config,
            neuron_core_pre_assignment=clustering.cluster_assignment.astype(np.int64),
            neuron_idx_pre_assignment=neuron_local_assignment.astype(np.int64),
            neuron_slice_assignment=neuron_slice_assignment.astype(np.int64)
        )
        state = MosaicNetworkMappingState(
            _mapping_input=mapping_input,
            assignment=assignment
        )
        return state

