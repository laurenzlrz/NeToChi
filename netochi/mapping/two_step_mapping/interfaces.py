

from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

from netochi.input_generator.interfaces import MappingInput


@dataclass
class ClusterOutput:
    labels: Dict[int, int]      # Node ID -> Cluster ID

@dataclass
class HierarchicalClusterOutput(ClusterOutput):
    cluster_parent: Dict[int, int]      # Cluster ID -> Parent Cluster ID (-1 for root)


class Clusterer(ABC):

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterOutput:
        pass