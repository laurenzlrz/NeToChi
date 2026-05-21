from typing import Any

from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterer, LocalAddressAssigner, SliceAssigner, \
    ClusterAndHwOutput, HwClusterer

from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput

from netochi.mapping.interfaces import MosaicHWMappingState, BaseMapper
from netochi.input_generator.interfaces import MappingInput



class ThreeStepMapper(BaseMapper[MosaicHWMappingState, MappingInput]):
    """
    Given mapping input, infers mapping + hardware
    """

    def __init__(self, clusterer: HwClusterer, address_assigner: LocalAddressAssigner, slice_assigner: SliceAssigner):
        self._clusterer = clusterer
        self._address_assigner = address_assigner
        self._slice_assigner = slice_assigner


    def run(self, mapping_input: MappingInput) -> MosaicHWMappingState:
        graph = mapping_input.graph

        # --- 1. Clustering ---
        clustering: ClusterAndHwOutput = self._clusterer.cluster(input_data=mapping_input)

        # --- 2. Local address assignment ---
        neuron_local_assignment = self._address_assigner.assign_addresses(graph=graph, clustering=clustering)

        # --- 3. Slice assignment ---
        neuron_slice_assignment = self._slice_assigner.assign_slices(clustering=clustering, graph=graph, local_assignment=neuron_local_assignment)

        # --- 4. Create Mapping State ---
        state = MosaicHWMappingState.create_uninitialized_state(mapping_input=mapping_input, initial_hw_guess=clustering.hw)
        state.neuron_slice_assignments = neuron_slice_assignment
        state.neuron_local_idxs_assignment = neuron_local_assignment
        state.neuron_core_idxs_assignment = clustering.cluster_assignment
        return state

