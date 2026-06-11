
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from netochi.input_generator.interfaces import MappingInput, MosaicHWMappingInput, HWMappingInput

import graph_tool as gt

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# =============== Cluster Output ====================

@dataclass
class ClusterOutput:
    cluster_assignment: npt.NDArray[np.int_]    # Node ID -> Cluster ID
    num_clusters: int                           # number of clusters on LOWEST level (= nr cores)

@dataclass
class HierarchicalClusterOutput(ClusterOutput):
    cluster_parent: npt.NDArray[np.int_]        # Cluster ID -> Parent Cluster ID (-1 for root)

@dataclass
class ClusterAndHwOutput(ClusterOutput):
    hw: MosaicHardwareConfig

# =============== Clusterer ========================

class Clusterer(ABC):

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterOutput:
        pass


class HierarchicalClusterer(ABC):
    """
    infers a hierarchical clustering
    """

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:
        pass

class ClustererOutputsHw(Clusterer):
    """
    outputs a clustering that fits onto the given hardware
    """

    @abstractmethod
    def cluster(self, input_data: MosaicHWMappingInput) -> ClusterAndHwOutput:
        pass


class ClustererFixedHw(ClustererOutputsHw):
    """
    outputs a clustering that fits onto the given hardware
    """

    @abstractmethod
    def cluster(self, input_data: HWMappingInput) -> ClusterAndHwOutput:
        pass

class ClustererInferHw(ClustererOutputsHw):
    """
    outputs a clustering and the corresponding hardware
    Clustering needs to fit on the outputted hardware!!!
    """

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        pass

class ClusteringAdapter:
    """
    given a hierarchical clustering, it infers hardware and adapts clustering so that it fits the hardware
    """

    @abstractmethod
    def adapt_clustering(self, clustering: HierarchicalClusterOutput) -> ClusterAndHwOutput:
        pass

class ClusteringAdapterFixedHw:
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
    def assign_addresses(self, graph: gt.Graph, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_] :
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

