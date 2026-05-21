from typing import Dict, List
from collections import defaultdict

from netochi.input_generator.interfaces import MappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.clustering.hierarchical_community_detection.hcd_clusterer import HcdClusterer
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, ClusterAndHwOutput, \
    ClusterAndHwOutput, HwClusterer
from netochi.mapping.three_step_mapping.clustering.hierarchical_community_detection.hardware_tranform_utils import compute_core_sizes, \
    compute_children_count


class HcdHwClusterer(HcdClusterer, HwClusterer):

    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        # --- 1. Get base clustering ---
        clustering: HierarchicalClusterOutput = super().cluster(input_data=input_data)

        # --- 2. Extract Constraints ---
        cluster_sizes = compute_core_sizes(clustering)
        neurons_per_core = max(cluster_sizes.values()) if cluster_sizes else 1

        children_counts = compute_children_count(clustering)
        nodes_per_router = max(children_counts.values()) if children_counts else 1

        router_levels = self._compute_depth(clustering.cluster_parent)

        # Build Adjacency List
        tree = defaultdict(list)
        for child, parent in clustering.cluster_parent.items():
            tree[parent].append(child)

        new_cluster_assignment = {}
        global_cluster_counter = 0

        def pad_recursive(current_parents: List[int], current_level: int):
            nonlocal global_cluster_counter

            next_level_clusters = []

            for p in current_parents:
                existing_children = tree.get(p, [])

                # Every router MUST have exactly nodes_per_router children
                for i in range(nodes_per_router):
                    new_cid = global_cluster_counter
                    global_cluster_counter += 1
                    next_level_clusters.append(new_cid)

                    if i < len(existing_children):
                        # This is a REAL cluster branch
                        old_cid = existing_children[i]

                        if current_level == router_levels - 1:
                            # At the leaf (Core), assign ONLY original neurons
                            # clustering.labels is NodeID -> ClusterID
                            for neuron_id, cluster_id in clustering.cluster_assignment.items():
                                if cluster_id == old_cid:
                                    new_cluster_assignment[neuron_id] = new_cid
                        # If not leaf, the recursion continues to find the leaves
                    else:
                        # This is a DUMMY cluster branch.
                        # We don't assign any neurons here.
                        # The CID is created to keep the 'new_num_clusters' count correct.
                        pass

            # Continue down the symmetric tree
            if current_level < router_levels - 1:
                pad_recursive(next_level_clusters, current_level + 1)

        # Start from root (-1)
        roots = tree.get(-1, [])
        pad_recursive(roots, 0)

        # Construct HW Config
        hw = MosaicHardwareConfig(
            nodes_per_router=nodes_per_router,
            neurons_per_core=neurons_per_core,
            router_levels=router_levels,
            slice_factor=2
        )

        return ClusterAndHwOutput(
            cluster_assignment=new_cluster_assignment,
            num_clusters=global_cluster_counter,  # Total physical cores (real + empty)
            hw=hw
        )

    def _compute_depth(self, cluster_parent: Dict[int, int]) -> int:
        """
        Computes the maximum depth (number of levels) of the hierarchy.
        The root's parent is -1.
        """

        def get_node_depth(node_id: int) -> int:
            # Base case: if we hit the root marker
            if node_id == -1:
                return 0

            parent_id = cluster_parent.get(node_id, -1)
            return 1 + get_node_depth(parent_id)

        return get_node_depth(0) # 0 is always leaf, every leaf has same depth