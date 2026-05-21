
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.interfaces import ClusteringAdapter, HierarchicalClusterOutput, \
    ClusterAndHwOutput
from netochi.mapping.three_step_mapping.clustering.hardware_cluster_adaptation_utils import compute_core_sizes, \
    compute_children_count, compute_hierarchy_depth, transform_hierarchy_into_adjacency_list

import math
from statistics import mean

class FlatRebuildClusteringAdapter(ClusteringAdapter):
    """
    this adapter throws away the hierarchy, takes the lowest-level cluster as leaves, and constructs a new hierarchy from scratch.
    The cores are grouped from left to right sorted by their indices.
    """

    def adapt_clustering(self, clustering: HierarchicalClusterOutput) -> ClusterAndHwOutput:

        # --- 1. Infer hw config ---
        cluster_sizes = compute_core_sizes(clustering)
        neurons_per_core = max(cluster_sizes.values())

        children_counts = compute_children_count(clustering)
        if clustering.num_clusters == 1:
            nodes_per_router = 1
        else:
            nodes_per_router = math.ceil(mean(children_counts.values())) # the mean

        if clustering.num_clusters <= 1:
            router_levels = 0
        else:
            router_levels = math.ceil(math.log(clustering.num_clusters, nodes_per_router))

        hw = MosaicHardwareConfig(
            nodes_per_router=nodes_per_router,
            neurons_per_core=neurons_per_core,
            router_levels=router_levels
        )

        new_num_clusters = nodes_per_router**router_levels  # Total nr of cores (real + dummy cores appended at the end)


        return ClusterAndHwOutput(
            cluster_assignment=clustering.cluster_assignment,
            num_clusters=new_num_clusters,
            hw=hw
        )


