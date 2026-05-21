
from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

from netochi.input_generator.interfaces import MappingInput

import graph_tool as gt

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# =============== Cluster Output ====================

@dataclass
class ClusterOutput:
    cluster_assignment: Dict[int, int]      # Node ID -> Cluster ID
    num_clusters: int

@dataclass
class HierarchicalClusterOutput(ClusterOutput):
    cluster_parent: Dict[int, int]      # Cluster ID -> Parent Cluster ID (-1 for root)

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


class HwClusterer(Clusterer):
    """
    infers a clustering and the corresponding hardware
    """

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        pass


# ============== Local Address Assigner =======================

class LocalAddressAssigner(ABC):
    """
    given a clustering and a hardware, assigns each neuron a local index within the core
    """

    @abstractmethod
    def assign_addresses(self, graph: gt.Graph, clustering: ClusterAndHwOutput) -> Dict[int, int]:
        """
        neuron_id -> local_idx
        """
        pass


# ============== Slice Assigner =======================

class SliceAssignerFlexibleHw(ABC):
    """
    given a clustering and a local address assignment, assigns each (neuron,dist) a slice index
    """
    # TODO check whether needed

    @abstractmethod
    def assign_slices(self, clustering: HierarchicalClusterOutput, graph: gt.Graph, local_assignment: Dict[int, int]) -> Dict[int, Dict[int, int]]:
        pass

class SliceAssigner(ABC):
    """
    given a clustering, a HARDWARE, and a local address assignment, assigns each (neuron,dist) a slice index
    """

    @abstractmethod
    def assign_slices(self, clustering: ClusterAndHwOutput, graph: gt.Graph, local_assignment: Dict[int, int]) -> Dict[int, Dict[int, int]]:
        pass

