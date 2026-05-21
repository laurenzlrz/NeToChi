from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.interfaces import ClusteringAdapter, HierarchicalClusterOutput, \
    ClusterAndHwOutput
from netochi.mapping.three_step_mapping.clustering.hardware_cluster_adaptation_utils import compute_core_sizes, \
    compute_children_count, compute_hierarchy_depth, transform_hierarchy_into_adjacency_list


class PaddingClusteringAdapter(ClusteringAdapter):

    def adapt_clustering(self, clustering: HierarchicalClusterOutput) -> ClusterAndHwOutput:

        # --- 1. Infer hw config ---
        cluster_sizes = compute_core_sizes(clustering)
        neurons_per_core = max(cluster_sizes.values()) if cluster_sizes else 1

        children_counts = compute_children_count(clustering)
        nodes_per_router = max(children_counts.values()) if children_counts else 1

        router_levels = compute_hierarchy_depth(clustering)

        hw = MosaicHardwareConfig(
            nodes_per_router=nodes_per_router,
            neurons_per_core=neurons_per_core,
            router_levels=router_levels
        )

        new_num_clusters = nodes_per_router**router_levels  # Total nr of cores (real + dummy)

        # --- 2. Pad and insert dummy neurons and cores: Compute the new cluster assignment ---
        # when inserting dummies, then the new cluster assignment needs to be adapted

        adj_list_clusters, map_leaf_to_neurons = transform_hierarchy_into_adjacency_list(clustering)
        new_cluster_assignment = [0] * len(clustering.cluster_assignment)

        # Identify the root cluster ID of the original hierarchy (where parent is -1)
        root_id = next(cid for cid, pid in clustering.cluster_parent.items() if pid == -1)

        def assign_hardware_ids(current_cluster_id: int, current_hw_id: int):
            # BASE CASE: If this cluster has no children, it's a leaf node (a physical core)
            if current_cluster_id not in adj_list_clusters or not adj_list_clusters[current_cluster_id]:
                # Fetch all original neurons belonging to this core
                neurons = map_leaf_to_neurons.get(current_cluster_id, [])
                # Map them directly to their newly calculated uniform hardware core ID
                for neuron_id in neurons:
                    new_cluster_assignment[neuron_id] = current_hw_id
                return

            # RECURSIVE STEP: Internal router node
            # We iterate over actual children. branch_idx handles the structural positioning.
            for branch_idx, child_id in enumerate(adj_list_clusters[current_cluster_id]):
                # Left-shift the parent ID in base-k and add the branch index
                child_hw_id = (current_hw_id * nodes_per_router) + branch_idx

                assign_hardware_ids(child_id, child_hw_id)

        # Start the top-down traversal from the root, assigning it a base hardware ID of 0
        assign_hardware_ids(root_id, current_hw_id=0)

        return ClusterAndHwOutput(
            cluster_assignment=new_cluster_assignment,
            num_clusters=new_num_clusters,
            hw=hw
        )


