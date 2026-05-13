

from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MosaicHWMappingState

import numpy as np

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


class SliceAssigner(ABC):

    @abstractmethod
    def assign_slices(self, input_data: MappingInput, clustering: ClusterOutput, core_sizes: np.array) -> MosaicHWMappingState:
        pass