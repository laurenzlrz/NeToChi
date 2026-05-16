import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt


def plot_clustering_comparison(
        g: gt.Graph,
        initial_assignment: np.ndarray,
        inferred_assignment: np.ndarray,
        filename: str,
        cluster_spread: float = 1.5
):
    """
    Visualizes graph clustering performance.
    - Spatial position represents the INITIAL (ground truth) cluster.
    - Node color represents the INFERRED cluster.
    """
    # 1. Initialize property maps for position and color
    pos = g.new_vertex_property("vector<double>")
    v_color = g.new_vertex_property("vector<double>")

    unique_initial = np.unique(initial_assignment)
    unique_inferred = np.unique(inferred_assignment)

    # 2. Determine centers for initial clusters (arrange them in a large circle)
    num_initial = len(unique_initial)
    # Scale radius based on number of clusters so they don't overlap
    radius = max(2.0, num_initial * cluster_spread * 0.3) # change scaling and max radius so update distance between clusters

    centers = {}
    for i, c_id in enumerate(unique_initial):
        theta = 2 * np.pi * i / num_initial
        centers[c_id] = (radius * np.cos(theta), radius * np.sin(theta))

    # 3. Assign node positions based on initial assignment
    for v in g.vertices():
        v_idx = int(v)
        c_init = initial_assignment[v_idx]
        cx, cy = centers[c_init]

        # Sample a random position within a circle of radius 'cluster_spread'
        r = cluster_spread * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()

        pos[v] = [cx + r * np.cos(theta), cy + r * np.sin(theta)]

    # 4. Assign node colors based on inferred assignment
    # Get a colormap from matplotlib (tab20 supports up to 20 distinct colors nicely)
    cmap = plt.get_cmap('tab20')
    inferred_to_idx = {c_id: i for i, c_id in enumerate(unique_inferred)}

    for v in g.vertices():
        v_idx = int(v)
        c_inf = inferred_assignment[v_idx]
        color_idx = inferred_to_idx[c_inf]

        # cmap returns an RGBA tuple, which graph-tool accepts natively
        v_color[v] = cmap(color_idx / max(1, len(unique_inferred) - 1))

    # 5. Draw and save the graph
    gt.graph_draw(
        g,
        pos=pos,
        vertex_fill_color=v_color,
        vertex_color="black",  # Border color for nodes
        vertex_size=8,
        edge_pen_width=0.8,
        edge_color=[0.5, 0.5, 0.5, 0.2],  # Semi-transparent gray edges to reduce visual clutter
        output=filename,
        output_size=(1000, 1000),
        fit_view=True
    )
    print(f"Plot saved to {filename}")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # A. Generate a mock graph (Stochastic Block Model)
    num_nodes = 300
    g = gt.Graph(directed=False)
    g.add_vertex(num_nodes)

    # B. Create Ground Truth (3 distinct clusters)
    initial_clusters = np.repeat([0, 1, 2], 100)

    # Add edges: dense inside clusters, sparse between them
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = 0.15 if initial_clusters[i] == initial_clusters[j] else 0.005
            if np.random.rand() < prob:
                g.add_edge(i, j)

    # C. Create Inferred Clustering (simulate some mistakes)
    inferred_clusters = initial_clusters.copy()
    # Scramble 10% of the nodes to simulate an imperfect clustering algorithm
    scramble_mask = np.random.rand(num_nodes) < 0.10
    inferred_clusters[scramble_mask] = np.random.choice([0, 1, 2], size=np.sum(scramble_mask))

    # D. Run visualization
    plot_clustering_comparison(
        g,
        initial_assignment=initial_clusters,
        inferred_assignment=inferred_clusters,
        filename="../results/clustering_evaluation.pdf"
    )