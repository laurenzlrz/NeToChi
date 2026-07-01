from pydantic import BaseModel, Field, ConfigDict

from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, SliceAssigner, \
    ClusterAndHwOutput, ClustererInferHw

from netochi.mapping.interfaces import MosaicHWMappingState, BaseMapper
from netochi.input_generator.interfaces import MappingInput, MosaicAssignment
import numpy as np

class ThreeStepMapperConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    clusterer: ClustererInferHw = Field(default=False)
    address_assigner: LocalAddressAssigner = Field(default=False)
    slice_assigner: SliceAssigner = Field(default=False)

    def create(self) -> "ThreeStepMapper":
        return ThreeStepMapper(self.clusterer, self.address_assigner, self.slice_assigner)


class ThreeStepMapper(BaseMapper[MosaicHWMappingState, MappingInput]):
    """
    Given mapping input, infers mapping + hardware
    """

    def __init__(self, clusterer: ClustererInferHw, address_assigner: LocalAddressAssigner, slice_assigner: SliceAssigner):
        self._clusterer = clusterer
        self._address_assigner = address_assigner
        self._slice_assigner = slice_assigner


    def run(self, mapping_input: MappingInput) -> MosaicHWMappingState:
        """
        the mapping runs in three stages:
            1. clustering: it outputs a clustering AND a hardware config. The clustering must match the hardware
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
            hw=clustering.hw,
            neuron_core_pre_assignment=clustering.cluster_assignment.astype(np.int64),
            neuron_idx_pre_assignment=neuron_local_assignment.astype(np.int64),
            neuron_slice_assignment=neuron_slice_assignment.astype(np.int64)
        )
        state = MosaicHWMappingState(
            mapping_input=mapping_input,
            inferred_hw=clustering.hw,
            assignment=assignment
        )
        return state

    def get_name(self) -> str:
        return f"3step_{self._clusterer.get_name()}_{self._address_assigner.get_name()}_{self._slice_assigner.get_name()}"
