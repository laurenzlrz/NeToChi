import graph_tool.all as gt
import numpy as np
from scipy import sparse
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.two_step_mapping.clustering.hierarchical_community_detection.cluster import Hierarchy, Partition
from netochi.mapping.two_step_mapping.clustering.hierarchical_community_detection.inference import infer_hierarchy
from netochi.mapping.two_step_mapping.interfaces import Clusterer, ClusterOutput, HierarchicalClusterOutput


class HcdClusterer(Clusterer):

    def cluster(self, input_data: MappingInput) -> HierarchicalClusterOutput:

        # --- Convert gt.Graph to Undirected Sparse Matrix ---
        A_directed = gt.adjacency(input_data.graph)
        A_symmetric = A_directed + A_directed.T
        A_symmetric.data = np.ones_like(A_symmetric.data)
        A_sparse = sparse.csr_matrix(A_symmetric)

        # --- Run Hierarchical Inference ---
        print("Starting Hierarchical Community Detection...")
        hierarchy = infer_hierarchy(A_sparse)

        # --- Convert to ClusterOutput
        return self._convert_to_hierarchical_output(hierarchy)

    def _convert_to_hierarchical_output(self, hierarchy: Hierarchy) -> HierarchicalClusterOutput:
        hierarchy.expand_partitions_to_full_graph() # for every level: list neuron_id -> cluster_id

        # 1. Extract Labels (Node ID -> Cluster ID): take the finest level (level 0) as the base labels
        finest_pvec = hierarchy[0].pvec_expanded
        labels = {int(node_id): int(cluster_id) for node_id, cluster_id in enumerate(finest_pvec)}

        # 2. Extract Cluster Hierarchy (Parent Mapping)
        cluster_offset = 0
        cluster_parent = {}
        for level_idx in range(1, len(hierarchy)):
            nr_clusters = len(hierarchy[level_idx].pvec)
            for child_id in range(nr_clusters):
                parent_id = hierarchy[level_idx].pvec[child_id] + cluster_offset + nr_clusters
                cluster_parent[child_id + cluster_offset] = parent_id
            cluster_offset += nr_clusters

        if len(hierarchy) == 1:
            num_clusters = 1
        else:
            num_clusters = len(hierarchy[1].pvec)
        return HierarchicalClusterOutput(cluster_assignment=labels, cluster_parent=cluster_parent, num_clusters=num_clusters)



