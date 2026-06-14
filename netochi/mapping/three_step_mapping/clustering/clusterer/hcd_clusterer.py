import graph_tool.all as gt
import numpy as np
from scipy import sparse
from typing import cast, Any
from scipy.sparse import csr_matrix
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.three_step_mapping.clustering.clusterer.utils_from_hierarchical_community_detection_paper.cluster import Hierarchy
from netochi.mapping.three_step_mapping.clustering.clusterer.utils_from_hierarchical_community_detection_paper.inference import infer_hierarchy
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, HierarchicalClusterer


class HcdClusterer(HierarchicalClusterer):

    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:

        # --- Convert gt.Graph to Undirected Sparse Matrix ---
        A_directed = cast(csr_matrix, sparse.csr_matrix(cast(Any, gt.adjacency(input_data.graph))))
        A_symmetric = cast(csr_matrix, A_directed + A_directed.T)
        A_symmetric.data = np.ones_like(A_symmetric.data)
        A_sparse = A_symmetric

        # --- Run Hierarchical Inference ---
        hierarchy = infer_hierarchy(A_sparse)

        # --- Convert to ClusterOutput ---
        return self._convert_to_hierarchical_output(hierarchy)

    def _convert_to_hierarchical_output(self, hierarchy: Hierarchy) -> HierarchicalClusterOutput:
        """
        The layout ouf the Hierarchy is as follows:
        - for every level l: hierarchy[l].pvec[id] -> parent cluster id
        - for level l=0: hierarchy[0].pvec[neuron_id] = cluster_id = core_id
        - len(hierarchy[l].pvec) = nr clusters on level l
        - hierarchy[l].pvec_expanded: only difference is that len(hierarchy[l].pvec_expanded) = nr_neurons (i.e. maps neuron_id -> cluster on level l)

        Therefore, we want to transform it into our HierarchicalCluster layout (which maps cluster_id -> parent_cluster_id)
        """

        # 1. Extract Labels (Node ID -> Cluster ID): take the finest level (level 0) as the base labels
        finest_pvec = hierarchy[0].pvec
        labels = np.array(finest_pvec, dtype=np.int_)

        # 2. Extract Cluster Hierarchy (Parent Mapping)
        cluster_offset = 0
        cluster_parent = np.array([], dtype=np.int_)
        for level_idx in range(1, len(hierarchy)): # skip level 0, because we only want the cluster hierarchy, without the neuron labels
            # for every level: for every child cluster on this level, infer parent cluster
            pvec = np.array(hierarchy[level_idx].pvec, dtype=np.int_)
            nr_child_clusters = len(pvec)
            level_parent_ids = pvec + cluster_offset + nr_child_clusters
            # Dynamically append/resize the array to include this level's parents
            cluster_parent = np.append(cluster_parent, level_parent_ids)
            # Update the running offset
            cluster_offset += nr_child_clusters

        # 3. Add root entry. check: if last hierarchy has more than one clusters, add additional root cluster
        unique_roots = set(hierarchy[len(hierarchy) - 1].pvec)
        if len(unique_roots) > 1:
            root_id = cluster_offset + len(unique_roots)
            extension_size = len(unique_roots) + 1  # elements for unique roots + 1 for the ultra-root
            cluster_parent = np.append(cluster_parent, np.full(extension_size, -1, dtype=np.int_))

            for unique in unique_roots:
                cluster_parent[cluster_offset + unique] = root_id

            cluster_parent[root_id] = -1
        else:
            cluster_parent = np.append(cluster_parent, -1)

        # 3. Computer number of clusters on the lowest level (= number of cores)
        num_clusters = len(set(hierarchy[0].pvec))

        return HierarchicalClusterOutput(cluster_assignment=labels, cluster_parent=cluster_parent, num_clusters=num_clusters)



