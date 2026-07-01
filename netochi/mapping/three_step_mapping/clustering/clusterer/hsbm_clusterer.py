import graph_tool.all as gt
import numpy as np
from graph_tool.inference import NestedBlockState
from scipy import sparse
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.three_step_mapping.clustering.clusterer.utils_from_hierarchical_community_detection_paper.cluster import Hierarchy
from netochi.mapping.three_step_mapping.clustering.clusterer.utils_from_hierarchical_community_detection_paper.inference import infer_hierarchy
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, HierarchicalClusterer


class HsbmClusterer(HierarchicalClusterer):

    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:
        nested_state: NestedBlockState = gt.minimize_nested_blockmodel_dl(input_data.graph)
        return self._convert_to_hierarchical_output(nested_state)

    def _convert_to_hierarchical_output(self, nested_state: NestedBlockState) -> HierarchicalClusterOutput:
        hierarchy = nested_state.get_bs()

        # ===== remap hierarchy: cluster indices are continuously from 0, 1, ..., X
        remapped_hierarchy = []
        active_old_ids = None

        for level_idx, pvec in enumerate(hierarchy):
            if level_idx == 0:
                # Base level: remap non-contiguous node labels to 0, 1, 2...
                # np.unique returns the active old IDs, and the new contiguous assignments
                unique_blocks, new_labels = np.unique(pvec, return_inverse=True)
                remapped_hierarchy.append(new_labels)
                active_old_ids = unique_blocks
            else:
                # Higher levels: extract the parent IDs, but ONLY for the active blocks
                # from the previous level.
                active_parents_old_ids = np.array(pvec)[active_old_ids]

                # Remap these parent IDs to new contiguous IDs for this level
                unique_parents, new_parents = np.unique(active_parents_old_ids, return_inverse=True)
                remapped_hierarchy.append(new_parents)
                active_old_ids = unique_parents

        # =====================================================================
        # Now we can safely build the output using the clean remapped_hierarchy
        # =====================================================================

        # 1. Extract Labels (Node ID -> Cluster ID): take the finest level (level 0)
        labels = np.array(remapped_hierarchy[0], dtype=np.int_)

        # 2. Extract Cluster Hierarchy (Parent Mapping)
        cluster_offset = 0
        cluster_parent = np.array([], dtype=np.int_)

        for level_idx in range(1, len(remapped_hierarchy)):
            pvec = remapped_hierarchy[level_idx]
            nr_child_clusters = len(pvec)
            level_parent_ids = pvec + cluster_offset + nr_child_clusters

            cluster_parent = np.append(cluster_parent, level_parent_ids)
            cluster_offset += nr_child_clusters

        # 3. Add root entry
        unique_roots = set(remapped_hierarchy[-1])
        if len(unique_roots) > 1:
            root_id = cluster_offset + len(unique_roots)
            extension_size = len(unique_roots) + 1  # elements for unique roots + 1 for the ultra-root
            cluster_parent = np.append(cluster_parent, np.full(extension_size, -1, dtype=np.int_))

            for unique in unique_roots:
                cluster_parent[cluster_offset + unique] = root_id

            cluster_parent[root_id] = -1
        else:
            cluster_parent = np.append(cluster_parent, -1)

        # 4. compute num_clusters
        num_clusters = len(set(remapped_hierarchy[0]))

        return HierarchicalClusterOutput(cluster_assignment=labels, cluster_parent=cluster_parent, num_clusters=num_clusters)



