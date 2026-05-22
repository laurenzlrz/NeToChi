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

        # 1. Extract Labels (Node ID -> Cluster ID): take the finest level (level 0) as the base labels
        finest_level = hierarchy[0]
        labels = np.array(finest_level, dtype=np.int_)

        # 2. Extract Cluster Hierarchy (Parent Mapping)
        cluster_offset = 0
        cluster_parent = np.array([], dtype=np.int_)
        for level_idx in range(1, len(hierarchy)): # skip level 0, because we only want the cluster hierarchy, without the neuron labels
            # for every level: for every child cluster on this level, infer parent cluster
            pvec = np.array(hierarchy[level_idx], dtype=np.int_)
            nr_child_clusters = len(pvec)
            level_parent_ids = pvec + cluster_offset + nr_child_clusters
            # Dynamically append/resize the array to include this level's parents
            cluster_parent = np.append(cluster_parent, level_parent_ids)
            # Update the running offset
            cluster_offset += nr_child_clusters


        # 3. Add root entry. check: if last hierarchy has more than one clusters, add additional root cluster
        unique_roots = set(hierarchy[len(hierarchy) - 1])
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
        num_clusters = nested_state.get_levels()[0].get_nonempty_B()

        return HierarchicalClusterOutput(cluster_assignment=labels, cluster_parent=cluster_parent, num_clusters=num_clusters)



