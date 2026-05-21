import graph_tool.all as gt
import numpy as np
from scipy import sparse
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.three_step_mapping.clustering.hierarchical_community_detection.utils_from_paper.cluster import Hierarchy
from netochi.mapping.three_step_mapping.clustering.hierarchical_community_detection.utils_from_paper.inference import infer_hierarchy
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput, HierarchicalClusterer


class HcdClusterer(HierarchicalClusterer):

    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:

        # --- Convert gt.Graph to Undirected Sparse Matrix ---
        A_directed = gt.adjacency(input_data.graph)
        A_symmetric = A_directed + A_directed.T
        A_symmetric.data = np.ones_like(A_symmetric.data)
        A_sparse = sparse.csr_matrix(A_symmetric)

        # --- Run Hierarchical Inference ---
        print("Starting Hierarchical Community Detection...")
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
        labels = {int(node_id): int(cluster_id) for node_id, cluster_id in enumerate(finest_pvec)}

        # 2. Extract Cluster Hierarchy (Parent Mapping)
        cluster_offset = 0
        cluster_parent = {}
        for level_idx in range(1, len(hierarchy)): # skip level 0, because we only want the cluster hierarchy, without the neuron labels
            # for every level: for every child cluster on this level, infer parent cluster
            nr_child_clusters = len(hierarchy[level_idx].pvec)
            for child_id in range(nr_child_clusters):
                # compute parent_id: add offset (represents the clusters already processed), add nr_clusters (offset between child and parent
                parent_id = hierarchy[level_idx].pvec[child_id] + cluster_offset + nr_child_clusters
                cluster_parent[child_id + cluster_offset] = parent_id
            cluster_offset += nr_child_clusters

        # check: last hierarchy has more than one: insert additional root
        unique_roots = set(hierarchy[len(hierarchy) - 1].pvec)
        if len(unique_roots) > 1:
            root_id = cluster_offset + len(unique_roots)
            for unique in unique_roots:
                cluster_parent[cluster_offset + unique] = root_id
            cluster_parent[root_id] = -1
        else:
            cluster_parent[cluster_offset] = -1 # parent of root is -1

        # 3. Computer number of clusters on the lowest level (= number of cores)
        num_clusters = len(set(hierarchy[0].pvec))

        return HierarchicalClusterOutput(cluster_assignment=labels, cluster_parent=cluster_parent, num_clusters=num_clusters)



