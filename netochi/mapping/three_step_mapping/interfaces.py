from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import icontract

from netochi.definitions.exceptions import DimensionError
from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput, HWMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.definitions.freezable import freezable, Freezable

import graph_tool as gt


# =============== Cluster Output ====================

@freezable
@icontract.invariant(lambda self: self.validate())
class ClusterOutput(Freezable):
    def __init__(self, cluster_assignment: npt.NDArray[np.int_], num_clusters: int) -> None:
        self.cluster_assignment = cluster_assignment
        self.num_clusters = num_clusters
        if self.__class__ is ClusterOutput:
            self.freeze()

    def validate(self) -> bool:
        if self.cluster_assignment.ndim != 1:
            raise DimensionError(f"cluster_assignment must be 1D and cluster_parent length {self.cluster_assignment.shape[0]} must match num_clusters {self.num_clusters}")
        higher_than_cluster_id = np.max(self.cluster_assignment) if self.cluster_assignment.size > 0 else -1
        if higher_than_cluster_id >= self.num_clusters:
            raise DimensionError(f"cluster_assignment contains cluster IDs higher than num_clusters. "
                                 f"Max cluster ID: {higher_than_cluster_id}, num_clusters: {self.num_clusters}")
        return True


@freezable
@icontract.invariant(lambda self: self.validate())
class HierarchicalClusterOutput(ClusterOutput):
    def __init__(self, cluster_assignment: npt.NDArray[np.int_], num_clusters: int, cluster_parent: npt.NDArray[np.int_]) -> None:
        self.cluster_parent = cluster_parent
        super().__init__(cluster_assignment, num_clusters)
        if self.__class__ is HierarchicalClusterOutput:
            self.freeze()

    def validate(self) -> bool:
        super().validate()
        if not (self.cluster_parent.ndim == 1 and len(self.cluster_parent) == self.num_clusters):
             raise DimensionError(f"cluster_parent must be 1D and cluster_parent length {self.cluster_parent.shape[0]} must match num_clusters {self.num_clusters}")
        return True


@freezable
@icontract.invariant(lambda self: self.validate())
class ClusterAndHwOutput(ClusterOutput):
    """
    the cluster assignment already fits onto hardware: cluster ids are incremented from left to right on hardware
    """
    def __init__(self, cluster_assignment: npt.NDArray[np.int_], num_clusters: int, hw: MosaicHardwareConfig) -> None:
        self.hw = hw
        super().__init__(cluster_assignment, num_clusters)
        if self.__class__ is ClusterAndHwOutput:
            self.freeze()

    def validate(self) -> bool:
        super().validate()
        if self.num_clusters > self.hw.total_cores:
            raise DimensionError(f"Number of clusters {self.num_clusters} exceeds total cores {self.hw.total_cores} in hardware config.")

        cluster_sizes = np.bincount(self.cluster_assignment, minlength=self.num_clusters) if self.cluster_assignment.size > 0 else np.array([], dtype=np.int_)
        max_cluster_size = np.max(cluster_sizes) if cluster_sizes.size > 0 else 0

        # Check against the hardware constraint
        if max_cluster_size > self.hw.neurons_per_core:
            raise DimensionError(
                f"Maximum cluster size ({max_cluster_size}) exceeds the hardware limit "
                f"of neurons per core ({self.hw.neurons_per_core})."
            )

        return True


# =============== Clusterer ========================

class Clusterer[ANY_MAPPING_INPUT: MappingInput](ABC):
    @abstractmethod
    def cluster(self, input_data: ANY_MAPPING_INPUT) -> ClusterOutput:
        pass


class HierarchicalClusterer(ABC):
    """
    infers a hierarchical clustering
    """
    @abstractmethod
    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:
        pass


class ClustererOutputsHw[ANY_MAPPING_INPUT: MappingInput](Clusterer[ANY_MAPPING_INPUT], ABC):
    """
    outputs a clustering that fits onto the given hardware
    """
    @abstractmethod
    def cluster(self, input_data: ANY_MAPPING_INPUT) -> ClusterAndHwOutput:
        pass


class ClustererFixedHw[WITH_HW_INPUT: HWMappingInput](ClustererOutputsHw[WITH_HW_INPUT], ABC):
    """
    outputs a clustering that fits onto the given hardware
    """
    @abstractmethod
    def cluster(self, input_data: WITH_HW_INPUT) -> ClusterAndHwOutput:
        pass


class ClustererInferHw(ClustererOutputsHw[MappingInput], ABC):
    """
    outputs a clustering and the corresponding hardware
    Clustering needs to fit on the outputted hardware!!!
    """
    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        pass


class ClusteringAdapter(ABC):
    """
    given a hierarchical clustering, it infers hardware and adapts clustering so that it fits the hardware
    """
    @abstractmethod
    def adapt_clustering(self, clustering: HierarchicalClusterOutput) -> ClusterAndHwOutput:
        pass


class ClusteringAdapterFixedHw(ABC):
    """
    given a hierarchical clustering, and hardware, it adapts clustering so that it fits the hardware
    """
    @abstractmethod
    def adapt_clustering(self, clustering: HierarchicalClusterOutput, hw_config: MosaicHardwareConfig) -> ClusterAndHwOutput:
        pass


# ============== Local Address Assigner =======================

class LocalAddressAssigner(ABC):
    """
    given a clustering and a hardware, assigns each neuron a local index within the core
    """
    @abstractmethod
    def assign_addresses(self, graph: gt.Graph, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:
        """
        neuron_id -> local_idx
        """
        pass


# ============== Slice Assigner =======================

class SliceAssigner(ABC):
    """
    given a clustering, a hardware, and a local address assignment, assigns each (neuron,dist) a slice index
    """
    @abstractmethod
    def assign_slices(self, clustering: ClusterAndHwOutput, graph: gt.Graph, local_assignment: npt.NDArray[np.int_]) -> np.ndarray[tuple[Any, Any], np.dtype[np.int_]]:
        pass
