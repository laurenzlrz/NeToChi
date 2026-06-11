
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, SliceAssigner, \
    ClusterAndHwOutput, ClustererFixedHw

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MappingInput, MosaicHWMappingInput


class ThreeStepHwMapper(BaseMapper[MosaicHWMappingInput, MappingInput]):
    """
    Given mapping input + hardware, infers mapping
    """

    def __init__(self, clusterer: ClustererFixedHw, address_assigner: LocalAddressAssigner, slice_assigner: SliceAssigner):
        self._clusterer = clusterer
        self._address_assigner = address_assigner
        self._slice_assigner = slice_assigner


    def run(self, mapping_input: MosaicHWMappingInput) -> MosaicNetworkMappingState:
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
        state = MosaicNetworkMappingState(mapping_input=mapping_input,
                                          neuron_slice_assignments = neuron_slice_assignment,
                                          neuron_local_idxs_assignment = neuron_local_assignment,
                                          neuron_core_idxs_assignment = clustering.cluster_assignment,
                                          model_config = None
                                          )
        return state

