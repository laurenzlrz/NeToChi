

import numpy as np
import networkx as nx
from netochi.network_generator.swta_generator import SwtaNetwork, SwtaGeneratorConfig
from scipy import sparse

from inference import infer_hierarchy


def run_swn_community_detection(config: SwtaGeneratorConfig):
    """
    Complete data flow: Generation -> Conversion -> Inference
    """
    # --- STEP 1: Generate the Network ---
    print(f"Generating sWTA with {config.total_neurons} neurons...")
    generator = SwtaNetwork(config)
    nx_digraph = generator.generate()

    # --- STEP 2: Convert DiGraph to Undirected Sparse Matrix ---
    A_directed = nx.to_scipy_sparse_array(nx_digraph, format='csr', dtype=float)
    A_symmetric = A_directed + A_directed.T
    A_symmetric.data = np.ones_like(A_symmetric.data) # entries either 1 or 0
    A_sparse = sparse.csr_matrix(A_symmetric)
    print(f"Conversion complete")

    # --- STEP 3: Run Hierarchical Inference ---
    print("Starting Hierarchical Community Detection...")
    hierarchy = infer_hierarchy(A_sparse)

    return hierarchy


if __name__ == "__main__":
    my_config = SwtaGeneratorConfig(
        num_clusters=10,
        neurons_per_cluster=30,
        seed=42,
        inhibitory_ratio=0.1
    )

    hierarchy_result = run_swn_community_detection(my_config)

    # Display results
    print("\nDetection Results:")
    for i, level in enumerate(hierarchy_result):
        print(f"Level {i}: Found {level.k} communities")