import itertools

import numpy as np
import graph_tool.all as gt

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.clustering.clusterer.kaHyPar_hyperedges_clusterer import \
    KaHyParHyperedgesClusterer
from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw
from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import KaMinParHierarchicalClusterer
from netochi.mapping.three_step_mapping.clustering.clusterer.kaHyPar_cluster import KaHyParHierarchicalClusterer
from netochi.visualization.visualize_clustering import plot_clustering_comparison


def get_clusterer() -> ClustererFixedHw:
    return KaHyParHyperedgesClusterer()
    #return KaMinParHierarchicalClusterer()
    #return KaHyParHierarchicalClusterer()

def test_hierarchical_clustering_structure():
    # 1. Setup the Hardware Config
    # 2 nodes per router, 2 router levels = 4 total cores.
    # 2 neurons per core limit.
    hw_config = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=2,
        router_levels=2,
        slice_factor=2
    )

    # 2. Construct a predictable hierarchical graph
    # We create 8 nodes. We want them paired: (0,1), (2,3), (4,5), (6,7).
    # We will connect the pairs with heavy intra-edges, and weak inter-edges.
    # Graph-tool doesn't need explicit weights if the topology clearly dictates the min-cut.
    graph = gt.Graph(directed=False)
    graph.add_vertex(8)

    edges = [
        # The 4 target core clusters (strong cliques)
        (0, 1), (2, 3), (4, 5), (6, 7),

        # Level 1 connections (will be cut in the 2nd bisection pass)
        (0, 2), (4, 6),

        # Level 0 root connection (will be cut in the 1st bisection pass)
        (2, 4)
    ]
    for src, dst in edges:
        graph.add_edge(src, dst)

    # 3. Setup Mapping Input
    # Using a placeholder dictionary for descriptions as defined in your dataclass
    mapping_input = MosaicMappingInput(
        graph=graph,
        descriptions={},
        hw_config=hw_config
    )

    # 4. Instantiate the Clusterer
    # We target 4 leaves to match the 4 available hardware cores
    clusterer = get_clusterer()

    # 5. Execute Clustering
    result = clusterer.cluster(mapping_input)

    # 6. Assertions

    # A. Validate Hardware Constraints and basic outputs
    assert result.num_clusters == 4, "Should have exactly 4 clusters matching the target leaves."
    assert result.cluster_assignment.shape == (8,), "Every node must have an assignment."

    # B. Validate the precise hierarchy groups (Structural Equivalence)
    # Regardless of left/right branch assignments in the tree, these specific
    # pairs of nodes MUST end up in the same hardware core.
    assert result.cluster_assignment[0] == result.cluster_assignment[1], "Nodes 0 and 1 should be clustered together."
    assert result.cluster_assignment[2] == result.cluster_assignment[3], "Nodes 2 and 3 should be clustered together."
    assert result.cluster_assignment[4] == result.cluster_assignment[5], "Nodes 4 and 5 should be clustered together."
    assert result.cluster_assignment[6] == result.cluster_assignment[7], "Nodes 6 and 7 should be clustered together."

    # C. Validate Hardware Capacity Restrictions
    # Ensure no core exceeds the `neurons_per_core=2` limit
    cluster_counts = np.bincount(result.cluster_assignment, minlength=result.num_clusters)
    for core_id, count in enumerate(cluster_counts):
        assert count <= hw_config.neurons_per_core, f"Core {core_id} exceeded maximum neuron capacity!"

    # D. Validate Continuous Left-to-Right Core IDs
    # Ensure all assigned IDs fall exactly within the range [0, 3]
    unique_clusters = np.unique(result.cluster_assignment)
    assert np.array_equal(unique_clusters, np.array([0, 1, 2, 3])), "Cluster IDs must be continuously mapped from 0 to 3."
    plot_clustering_comparison(
        graph,
        initial_assignment=np.array([0,0,1,1,2,2,3,3]),
        inferred_assignment=result.cluster_assignment,
        filename="../test_results/cluster_hierarchy_test_output.pdf"
    )


def test_kaminpar_clustering_16_neurons_4_cores():
    # 1. Setup the Hardware Config
    # 2 nodes per router, 2 router levels = 4 total cores.
    # Now supporting 4 neurons per core.
    hw_config = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=4,
        router_levels=2,
        slice_factor=2
    )

    # 2. Construct a predictable hierarchical graph of 16 nodes
    graph = gt.Graph(directed=False)
    graph.add_vertex(16)

    # Define our 4 expected clusters
    expected_clusters = [
        [0, 1, 2, 3],  # Target Core A
        [4, 5, 6, 7],  # Target Core B
        [8, 9, 10, 11],  # Target Core C
        [12, 13, 14, 15]  # Target Core D
    ]

    # Create dense cliques within each cluster to prevent internal cutting
    edges = []
    for group in expected_clusters:
        edges.extend(list(itertools.combinations(group, 2)))

    # Add weak inter-cluster bridge edges
    edges.extend([
        # Level 1 connections (will be cut in the 2nd bisection pass)
        (3, 4),  # Connects Core A to Core B
        (11, 12),  # Connects Core C to Core D

        # Level 0 root connection (will be cut in the 1st bisection pass)
        (7, 8)  # Connects the AB super-cluster to the CD super-cluster
    ])

    for src, dst in edges:
        graph.add_edge(src, dst)

    # 3. Setup Mapping Input
    mapping_input = MosaicMappingInput(
        graph=graph,
        descriptions={},
        hw_config=hw_config
    )

    # 4. Instantiate the Clusterer
    # Target 4 leaves to match the 4 available hardware cores
    clusterer = get_clusterer()

    # 5. Execute Clustering
    result = clusterer.cluster(mapping_input)

    # 6. Assertions

    # A. Validate Hardware Constraints and basic outputs
    assert result.num_clusters == 4, "Should have exactly 4 clusters matching the target leaves."
    assert result.cluster_assignment.shape == (16,), "Every node must have an assignment."

    # B. Validate the precise hierarchy groups (Structural Equivalence)
    # Ensure all nodes within a designed clique share the exact same assigned cluster ID
    for group in expected_clusters:
        first_node_id = result.cluster_assignment[group[0]]
        for node in group[1:]:
            assert result.cluster_assignment[node] == first_node_id, \
                f"Node {node} was separated from its clique {group}!"

    # C. Validate Hardware Capacity Restrictions
    cluster_counts = np.bincount(result.cluster_assignment, minlength=result.num_clusters)
    for core_id, count in enumerate(cluster_counts):
        assert count <= hw_config.neurons_per_core, \
            f"Core {core_id} has {count} neurons, exceeding maximum capacity of {hw_config.neurons_per_core}!"

    # D. Validate Continuous Left-to-Right Core IDs
    unique_clusters = np.unique(result.cluster_assignment)
    assert np.array_equal(unique_clusters, np.array([0, 1, 2, 3])), \
        "Cluster IDs must be continuously mapped from 0 to 3."

    plot_clustering_comparison(
        graph,
        initial_assignment=np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int_),
        inferred_assignment=result.cluster_assignment,
        filename="../test_results/cluster_hierarchy_test_output.pdf"
    )


def test_kaminpar_clustering_27_neurons_9_cores_3_ary():
    # 1. Setup the Hardware Config
    # 3 nodes per router, 2 router levels = 3^2 = 9 total cores.
    # 3 neurons per core = 27 total nodes.
    hw_config = MosaicHardwareConfig(
        nodes_per_router=3,
        neurons_per_core=3,
        router_levels=2,
        slice_factor=3
    )

    # 2. Construct a predictable hierarchical graph of 27 nodes
    graph = gt.Graph(directed=False)
    graph.add_vertex(27)

    # Define our 9 expected clusters (3 nodes each)
    expected_clusters = [
        [0, 1, 2],  # Core 0 (Super-cluster 0)
        [3, 4, 5],  # Core 1 (Super-cluster 0)
        [6, 7, 8],  # Core 2 (Super-cluster 0)

        [9, 10, 11],  # Core 3 (Super-cluster 1)
        [12, 13, 14],  # Core 4 (Super-cluster 1)
        [15, 16, 17],  # Core 5 (Super-cluster 1)

        [18, 19, 20],  # Core 6 (Super-cluster 2)
        [21, 22, 23],  # Core 7 (Super-cluster 2)
        [24, 25, 26]  # Core 8 (Super-cluster 2)
    ]

    edges = []
    # Create dense cliques within each cluster to prevent internal cutting
    for group in expected_clusters:
        edges.extend(list(itertools.combinations(group, 2)))

    # Add weak inter-cluster bridge edges to define the hierarchy
    edges.extend([
        # Level 1 connections (will be cut in the 2nd pass: separating the 3 super-clusters into 9 cores)
        (2, 3),  # Connects Core 0 to Core 1
        (5, 6),  # Connects Core 1 to Core 2
        (11, 12),  # Connects Core 3 to Core 4
        (14, 15),  # Connects Core 4 to Core 5
        (20, 21),  # Connects Core 6 to Core 7
        (23, 24),  # Connects Core 7 to Core 8

        # Level 0 root connections (will be cut in the 1st pass: separating into 3 super-clusters)
        (8, 9),  # Connects Super-cluster 0 to Super-cluster 1
        (17, 18)  # Connects Super-cluster 1 to Super-cluster 2
    ])

    for src, dst in edges:
        graph.add_edge(src, dst)

    # 3. Setup Mapping Input
    mapping_input = MosaicMappingInput(
        graph=graph,
        descriptions={},
        hw_config=hw_config
    )

    # 4. Instantiate the Clusterer
    # Target 9 leaves to match the 9 available hardware cores
    clusterer = get_clusterer()

    # 5. Execute Clustering
    result = clusterer.cluster(mapping_input)

    # 6. Assertions

    # A. Validate Hardware Constraints and basic outputs
    assert result.num_clusters == 9, "Should have exactly 9 clusters matching the target leaves."
    assert result.cluster_assignment.shape == (27,), "Every node must have an assignment."

    # B. Validate the precise hierarchy groups (Structural Equivalence)
    for group in expected_clusters:
        first_node_id = result.cluster_assignment[group[0]]
        for node in group[1:]:
            assert result.cluster_assignment[node] == first_node_id, \
                f"Node {node} was separated from its clique {group}!"

    # C. Validate Hardware Capacity Restrictions
    cluster_counts = np.bincount(result.cluster_assignment, minlength=result.num_clusters)
    for core_id, count in enumerate(cluster_counts):
        assert count <= hw_config.neurons_per_core, \
            f"Core {core_id} has {count} neurons, exceeding maximum capacity of {hw_config.neurons_per_core}!"

    # D. Validate Continuous Core IDs (0 through 8)
    unique_clusters = np.unique(result.cluster_assignment)
    assert np.array_equal(unique_clusters, np.arange(9)), \
        "Cluster IDs must be continuously mapped from 0 to 8."

    # E. Visualization
    # Create the expected initial assignment [0,0,0, 1,1,1, ..., 8,8,8]
    initial_assignment = np.repeat(np.arange(9), 3).astype(np.int_)

    plot_clustering_comparison(
        graph,
        initial_assignment=initial_assignment,
        inferred_assignment=result.cluster_assignment,
        filename="../test_results/cluster_hierarchy_3ary_test_output.pdf"
    )