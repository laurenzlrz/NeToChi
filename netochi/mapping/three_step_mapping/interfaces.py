
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from hamcrest.core import description
from pydantic import BaseModel, Field, model_validator, ConfigDict

from netochi.definitions.exceptions import DimensionError
from netochi.input_generator.interfaces import MappingInput, MosaicHWMappingInput, HWMappingInput

import graph_tool as gt

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


# =============== Cluster Output ====================

@dataclass
class ClusterOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    cluster_assignment: npt.NDArray[np.int_] = Field(description="Node ID -> Cluster/Core ID")
    num_clusters: int = Field(ge=0, description="number of clusters on LOWEST level (= nr cores)")

    @model_validator(mode="after")
    def validate_hierarchy(self: "ClusterOutput") -> "ClusterOutput":
        if not (self.cluster_assignment.ndim == 1):
           raise DimensionError(f"cluster_assignment must be 1D and cluster_parent length {self.cluster_assignment.shape[0]} must match num_clusters {self.num_clusters}")
        higher_than_cluster_id = np.max(self.cluster_assignment)
        if higher_than_cluster_id >= self.num_clusters:
            raise DimensionError(f"cluster_assignment contains cluster IDs higher than num_clusters. "
                                 f"Max cluster ID: {higher_than_cluster_id}, num_clusters: {self.num_clusters}")
        return self

@dataclass
class HierarchicalClusterOutput(ClusterOutput):
    cluster_parent: npt.NDArray[np.int_] = Field(description="Cluster ID -> Parent Cluster ID (-1 for root)")

    @model_validator(mode="after")
    def validate_hierarchy(self: "HierarchicalClusterOutput") -> "HierarchicalClusterOutput":
        if not (self.cluster_parent.ndim == 1 and len(self.cluster_parent) == self.num_clusters):
            raise DimensionError(f"cluster_parent must be 1D and cluster_parent length {self.cluster_parent.shape[0]} must match num_clusters {self.num_clusters}")
        return self

@dataclass
class ClusterAndHwOutput(ClusterOutput):
    hw: MosaicHardwareConfig = Field(description="Hardware configuration that the clustering fits on")

    @model_validator(mode="after")
    def validate_hardware_fit(self: "ClusterAndHwOutput") -> "ClusterAndHwOutput":
        if self.num_clusters > self.hw.total_cores:
            raise DimensionError(f"Number of clusters {self.num_clusters} exceeds total cores {self.hw.total_cores} in hardware config.")

        cluster_sizes = np.bincount(self.cluster_assignment, minlength=self.num_clusters)

        # Find the largest cluster size
        max_cluster_size = np.max(cluster_sizes)

        # Check against the hardware constraint
        if max_cluster_size > self.hw.neurons_per_core:
            raise DimensionError(
                f"Maximum cluster size ({max_cluster_size}) exceeds the hardware limit "
                f"of neurons per core ({self.hw.neurons_per_core})."
            )

        return self


# =============== Clusterer ========================

class Clusterer(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterOutput:
        pass


class HierarchicalClusterer(BaseModel, ABC):
    """
    infers a hierarchical clustering
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:
        pass

class ClustererOutputsHw(BaseModel, Clusterer):
    """
    outputs a clustering that fits onto the given hardware
    """

    @abstractmethod
    def cluster(self, input_data: MosaicHWMappingInput) -> ClusterAndHwOutput:
        pass


class ClustererFixedHw(BaseModel, ClustererOutputsHw):
    """
    outputs a clustering that fits onto the given hardware
    """

    @abstractmethod
    def cluster(self, input_data: MosaicHWMappingInput) -> ClusterAndHwOutput:
        pass

class ClustererInferHw(BaseModel, ClustererOutputsHw):
    """
    outputs a clustering and the corresponding hardware
    Clustering needs to fit on the outputted hardware!!!
    """

    @abstractmethod
    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        pass

class ClusteringAdapter(BaseModel, ):
    """
    given a hierarchical clustering, it infers hardware and adapts clustering so that it fits the hardware
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def adapt_clustering(self, clustering: HierarchicalClusterOutput) -> ClusterAndHwOutput:
        pass

class ClusteringAdapterFixedHw(BaseModel, ):
    """
    given a hierarchical clustering, and hardware, it adapts clustering so that it fits the hardware
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def adapt_clustering(self, clustering: HierarchicalClusterOutput, hw_config: MosaicHardwareConfig) -> ClusterAndHwOutput:
        pass

# ============== Local Address Assigner =======================

class LocalAddressAssigner(BaseModel, ABC):
    """
    given a clustering and a hardware, assigns each neuron a local index within the core
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def assign_addresses(self, graph: gt.Graph, clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_] :
        """
        neuron_id -> local_idx
        """
        pass


# ============== Slice Assigner =======================

class SliceAssigner(BaseModel, ABC):
    """
    given a clustering, a hardware, and a local address assignment, assigns each (neuron,dist) a slice index
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    @abstractmethod
    def assign_slices(self, clustering: ClusterAndHwOutput, graph: gt.Graph, local_assignment: npt.NDArray[np.int_]) -> np.ndarray[tuple[Any, Any], np.dtype[np.int_]]:
        pass

