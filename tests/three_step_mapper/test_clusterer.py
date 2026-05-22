import graph_tool.all as gt
import numpy as np
import os

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.three_step_mapping.clustering.clusterer.hcd_clusterer import HcdClusterer
from netochi.mapping.three_step_mapping.interfaces import HierarchicalClusterOutput
from netochi.visualization.visualize_clustering import plot_clustering_comparison


# --- Helper Function ---
def run_clusterer(graph: gt.Graph) -> HierarchicalClusterOutput:
    """
    Runs the HcdClusterer on the provided graph.
    """
    input_data = MappingInput(graph=graph, descriptions={}, payload=None)
    return HcdClusterer().cluster(input_data) # configure the desired clusterer here!!


# --- Test Suite ---
class TestClusterer:
    """
    tests any clusterer. Set the desired clusterer in the run_clusterer function above.
    """

    def test_four_ten_neuron_clusters(self):
        """
        Tests 4 distinct clusters containing 10 neurons each.
        The clusters are densely connected internally and weakly linked in a chain.
        """
        g = gt.Graph(directed=False)

        cluster_size = 10
        num_clusters = 4
        total_nodes = cluster_size * num_clusters
        g.add_vertex(total_nodes)

        # 1. Create 4 dense internal communities (cliques)
        for c in range(num_clusters):
            start_idx = c * cluster_size
            for i in range(cluster_size):
                for j in range(i + 1, cluster_size):
                    g.add_edge(start_idx + i, start_idx + j)

        # 2. Add weak bridges to link them together in a sequential chain:
        # Cluster 0 <-> Cluster 1 <-> Cluster 2 <-> Cluster 3
        for c in range(num_clusters - 1):
            current_cluster_last_node = (c * cluster_size) + (cluster_size - 1)
            next_cluster_first_node = (c + 1) * cluster_size
            g.add_edge(current_cluster_last_node, next_cluster_first_node)

        # 3. Dynamic Ground Truth Setup
        ground_truth_assignment = np.repeat(np.arange(num_clusters), cluster_size)

        output = run_clusterer(g)

        # 4. Invariant Testing: Ensure distinct internal blocks are tracked uniformly
        assign = output.cluster_assignment
        assert output.num_clusters == 4, f"Expected 4 clusters, got {output.num_clusters}"
        assert len(assign) == total_nodes

        # Verify that nodes within the same block share identical cluster IDs
        for c in range(num_clusters):
            offset = c * cluster_size
            base_cluster_id = assign[offset]
            for step in range(cluster_size):
                assert assign[offset + step] == base_cluster_id, (
                    f"Node {offset + step} split from its expected community block {c}."
                )

    def test_larger_graph_visualization(self):
        """
        Generates a larger graph consisting of 4 distinct communities (cliques)
        loosely connected in a ring.
        Plots the clustering and saves it in the test_results folder.
        """
        g = gt.Graph(directed=False)

        clique_size = 8
        num_cliques = 4
        total_nodes = clique_size * num_cliques
        g.add_vertex(total_nodes)

        # 1. Create dense cliques
        for c in range(num_cliques):
            start_idx = c * clique_size
            for i in range(clique_size):
                for j in range(i + 1, clique_size):
                    g.add_edge(start_idx + i, start_idx + j)

        # 2. Connect the cliques in a ring with single edges
        for c in range(num_cliques):
            current_clique_node = (c * clique_size) + (clique_size - 1)
            next_clique_node = ((c + 1) % num_cliques) * clique_size
            g.add_edge(current_clique_node, next_clique_node)

        # 3. Generate Ground Truth Assignment
        # This creates an array: [0,0..0, 1,1..1, 2,2..2, 3,3..3]
        ground_truth_assignment = np.repeat(np.arange(num_cliques), clique_size)

        # Run clustering
        output = run_clusterer(g)

        # Assert basic integrity before visualizing
        assert len(output.cluster_assignment) == total_nodes
        print(f"\n[CWD] Current Working Directory is: {os.getcwd()}")
        # 4. Visualize and compare
        plot_clustering_comparison(
            g,
            initial_assignment=ground_truth_assignment,
            inferred_assignment=output.cluster_assignment,
            filename="../test_results/cluster_test_output.pdf"
        )