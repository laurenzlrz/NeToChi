from typing import Dict, List
from collections import defaultdict

from mypy.solve import compute_dependencies

from netochi.input_generator.interfaces import MappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.clustering.hierarchical_community_detection.hcd_clusterer import HcdClusterer
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, ClusterAndHwOutput, \
    ClusterAndHwOutput, HwClusterer
from netochi.mapping.three_step_mapping.slice_assignment.slice_assignment_utils import compute_core_sizes, \
    compute_children_count


class HcdHwClusterer(HcdClusterer, HwClusterer):

    def cluster(self, input_data: MappingInput) -> ClusterAndHwOutput:
        # --- 1. Get base clustering ---
        clustering: HierarchicalClusterOutput = super().cluster(input_data=input_data)

        # --- 2. Construct Routing Hierarchy with uniform neurons per core and children per router ---
        # compute parameters
        cluster_sizes = compute_core_sizes(clustering)
        neurons_per_core = max(cluster_sizes.values()) if cluster_sizes else 1

        children_counts = compute_children_count(clustering)
        nodes_per_router = max(children_counts.values()) if children_counts else 1

        # compute new cluster assignment: pad with dummy nodes if there are not enough routing children
        router_levels = self._compute_depth(clustering.cluster_parent)
        new_cluster_assignment = {}

        # todo: compute new_num_clusters and new_cluster_assignment
        # Build Adjacency List for Traversal
        tree = defaultdict(list)
        for child, parent in clustering.cluster_parent.items():
            tree[parent].append(child)

        # --- 4. Padding Logic ---
        new_cluster_assignment = {}
        global_cluster_counter = 0
        global_neuron_counter = 0

        # Maps original Cluster ID -> New HW Cluster ID
        cluster_map: Dict[int, int] = {}

        def pad_recursive(current_parents: List[int], current_level: int):
            nonlocal global_cluster_counter, global_neuron_counter

            next_level_clusters = []

            for p in current_parents:
                # Get existing children from the base clustering
                existing_children = tree.get(p, [])

                # Fill up to nodes_per_router
                for i in range(nodes_per_router):
                    new_cid = global_cluster_counter
                    global_cluster_counter += 1
                    next_level_clusters.append(new_cid)

                    if i < len(existing_children):
                        # This is a real cluster
                        old_cid = existing_children[i]
                        cluster_map[old_cid] = new_cid

                        if current_level == router_levels - 1:
                            # We are at the leaf (Core) level, assign real neurons
                            original_neurons = [n for n, c in clustering.labels.items() if c == old_cid]
                            for neuron_id in original_neurons:
                                new_cluster_assignment[neuron_id] = new_cid

                            # Pad core with dummy neurons to reach neurons_per_core
                            for _ in range(neurons_per_core - len(original_neurons)):
                                # Negative IDs or high offsets often denote dummy neurons
                                dummy_nid = f"dummy_n_{global_neuron_counter}"
                                new_cluster_assignment[dummy_nid] = new_cid
                                global_neuron_counter += 1
                    else:
                        # This is a DUMMY cluster (Empty Core/Router)
                        if current_level == router_levels - 1:
                            # Pad a completely empty core with dummy neurons
                            for _ in range(neurons_per_core):
                                dummy_nid = f"dummy_n_{global_neuron_counter}"
                                new_cluster_assignment[dummy_nid] = new_cid
                                global_neuron_counter += 1

            # Continue down if not at the bottom
            if current_level < router_levels - 1:
                pad_recursive(next_level_clusters, current_level + 1)

        # Start recursion from root clusters (those parented by -1)
        roots = tree.get(-1, [])
        # Ensure roots also respect the nodes_per_router constraint if applicable,
        # or just start from the top-most found nodes.
        pad_recursive(roots, 0)

        new_num_clusters = global_cluster_counter


        # --- Construct HW Config ---
        hw = MosaicHardwareConfig(
            nodes_per_router=nodes_per_router,
            neurons_per_core=neurons_per_core,
            router_levels=router_levels,
            slice_factor=2
        )

        return ClusterAndHwOutput(
            cluster_assignment=new_cluster_assignment,
            num_clusters=new_num_clusters,
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