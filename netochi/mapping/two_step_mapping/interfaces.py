
from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

from netochi.input_generator.interfaces import MappingInput

import graph_tool as gt


@dataclass
class ClusterOutput:
    cluster_assignment: Dict[int, int]      # Node ID -> Cluster ID
    num_clusters: int

@dataclass
class HierarchicalClusterOutput(ClusterOutput):
    cluster_parent: Dict[int, int]      # Cluster ID -> Parent Cluster ID (-1 for root)


class Clusterer(ABC):

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterOutput:
        pass


class HierarchicalClusterer(ABC):

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:
        pass


class SliceAssigner(ABC):

    @abstractmethod
    def assign_slices(self, clustering: HierarchicalClusterOutput, graph: gt.Graph) -> Dict[int, Dict[int, int]]:
        pass


class LocalAddressAssigner(ABC):

    @abstractmethod
    def assign_addresses(self, graph: gt.Graph, clustering: HierarchicalClusterOutput) -> Dict[int, int]:
        """
        neuron_id -> local_idx
        """
        pass
