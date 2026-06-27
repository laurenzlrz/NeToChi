import itertools

import numpy as np
import graph_tool.all as gt

from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import KaMinParHierarchicalClusterer
from netochi.visualization.visualize_clustering import plot_clustering_comparison


def test_kaminpar_hierarchical_clustering_structure():
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
    clusterer = KaMinParHierarchicalClusterer()

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
    clusterer = KaMinParHierarchicalClusterer()

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
