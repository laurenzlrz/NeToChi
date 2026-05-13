
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterer, LocalAddressAssigner, SliceAssigner

from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput

from netochi.mapping.interfaces import MosaicHWMappingState, BaseMapper
from netochi.input_generator.interfaces import MappingInput


# TODO does not work yet: no flexible hw config implemented yet
class ThreeStepMapperFullyFlexibleHw(BaseMapper[MosaicHWMappingState, MappingInput]):
    """
    assumption: the hardware is flexible
    """

    def __init__(self, clusterer: HierarchicalClusterer, address_assigner: LocalAddressAssigner, slice_assigner: SliceAssigner):
        self.clusterer = clusterer
        self.address_assigner = address_assigner
        self.slice_assigner = slice_assigner


    def run(self, mapping_input: MappingInput) -> MosaicHWMappingState:
        graph = mapping_input.graph

        # --- 1. clustering ---
        clustering: HierarchicalClusterOutput = self.clusterer.cluster(input_data=mapping_input)

        # --- 2. Local address assignment ---
        neuron_local_assignment = self.address_assigner.assign_addresses(graph=graph, clustering=clustering)

        # --- 3. Slice assignment ---
        neuron_slice_assignment = self.slice_assigner.assign_slices(clustering=clustering, graph=graph, local_assignment=neuron_local_assignment)

        # todo create hw config
        state = MosaicHWMappingState.create_uninitialized_state(mapping_input=mapping_input, initial_hw_guess=hw_config)
        state.neuron_slice_assignments = neuron_slice_assignment
        state.neuron_local_idxs_assignment = neuron_local_assignment
        state.neuron_core_idxs_assignment = clustering.cluster_assignment
        return state

