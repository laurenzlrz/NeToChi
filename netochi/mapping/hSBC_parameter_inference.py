import graph_tool.all as gt
import numpy as np
from scipy.optimize import root


def generate_verification_graph(n0, b, H, p0):
    """
    Generates a hierarchical SBM directed graph using clear logic.
    Evaluates all ordered node pairs (i, j) for i != j to apply the p0 * 2^-h rule.
    """
    # Total nodes: N = n0 * b^H
    N = int(n0 * (b ** H))
    print(f"[1/3] Initializing directed graph with N = {N} vertices (n0={n0}, b={b}, H={H})...")

    # Set directed=True for directed graph mapping
    g = gt.Graph(directed=True)
    g.add_vertex(N)
    shuffled_lookup = np.random.permutation(N)

    print("[2/3] Computing hierarchical directed edges (this might take a few seconds)...")

    # In a directed graph, we evaluate all N * (N - 1) permutations
    total_pairs = N * (N - 1)
    pairs_processed = 0
    checkpoint = max(1, total_pairs // 10)

    # Iterate through all ordered pairs (i, j) where i != j
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Find the Lowest Common Ancestor (LCA) distance 'h'
            if (i // n0) == (j // n0):
                # Nodes belong to the same base cluster
                h = 0
            else:
                # Find the tree layer where their cluster paths diverge
                cluster_i = i // n0
                cluster_j = j // n0
                h = 1
                while (cluster_i // b) != (cluster_j // b):
                    cluster_i //= b
                    cluster_j //= b
                    h += 1

            # Compute connection probability: p_h = p0 * (1 / 2)^h
            p_h = p0 * (0.5 ** h)

            # Sample the directed edge independently
            if np.random.rand() < p_h:
                shuffled_i = shuffled_lookup[i]
                shuffled_j = shuffled_lookup[j]
                g.add_edge(shuffled_i, shuffled_j)

            pairs_processed += 1
            if pairs_processed % checkpoint == 0:
                progress = (pairs_processed / total_pairs) * 100
                print(f"      Progress: {progress:.0f}% of ordered pairs evaluated.")

    print(f"[3/3] Graph generation complete. Total Directed Edges: {g.num_edges()}")
    return g


def extract_graph_moments(g):
    """
    Extracts empirical moments from the directed graph using high-performance vector interfaces.
    """
    # 1. Extract total degrees (in-degree + out-degree)
    degrees = g.get_total_degrees()
    mean_d = np.mean(degrees)
    var_d = np.var(degrees, ddof=1)

    # 2. Determine local clustering coefficients based on the underlying undirected skeleton
    c_v = gt.local_clustering(g, undirected=True).a

    # Reconstruct the number of triangles per node
    triangles = 0.5 * c_v * degrees * (degrees - 1)

    # Eliminate NaN values caused by nodes with total degree < 2
    triangles = np.nan_to_num(triangles)
    mean_t = np.mean(triangles)

    return mean_d, var_d, mean_t


def theoretical_moments(params, N):
    """
    Computes the exact mathematical expected values of the hSBM for a directed graph setup.
    params = [n0, b, p0]
    """
    n0, b, p0 = params

    # Guard boundaries to protect the root-finding numerical solver
    if n0 < 2 or b <= 1 or p0 <= 0 or p0 > 1:
        return np.array([1e6, 1e6, 1e6])

    H = np.log(N / n0) / np.log(b)
    H_int = int(np.clip(np.round(H), 1, 20))  # Stable summation boundary

    # --- 1. Expected Total Degree E[D] ---
    # Multiplied by 2 because in-degree and out-degree are symmetric and independent
    E_D = 2.0 * (n0 - 1) * p0
    for h in range(1, H_int + 1):
        delta_N = n0 * (b ** (h - 1)) * (b - 1)
        E_D += 2.0 * delta_N * p0 * (2.0 ** (-h))

    # --- 2. Expected Degree Variance Var(D) ---
    # Multiplied by 2 due to the sum of independent in/out variance contributions
    Var_D = 2.0 * (n0 - 1) * p0 * (1.0 - p0)
    for h in range(1, H_int + 1):
        delta_N = n0 * (b ** (h - 1)) * (b - 1)
        p_h = p0 * (2.0 ** (-h))
        Var_D += 2.0 * delta_N * p_h * (1.0 - p_h)

    # --- 3. Expected Triangles E[T] ---
    # Evaluated on the underlying undirected triplet configurations
    E_T_intra = 0.5 * (n0 - 1) * (n0 - 2) * (p0 ** 3)

    E_T_mixed = 0.0
    for h in range(1, H_int + 1):
        delta_N = n0 * (b ** (h - 1)) * (b - 1)
        E_T_mixed += (n0 - 1) * delta_N * (p0 ** 3) * (4.0 ** (-h))

    E_T_inter = 0.0
    # Asymmetric inter-cluster contributions
    for h in range(2, H_int + 1):
        delta_N_h = n0 * (b ** (h - 1)) * (b - 1)
        for m in range(1, h):
            delta_N_m = n0 * (b ** (m - 1)) * (b - 1)
            E_T_inter += 2.0 * delta_N_m * delta_N_h * (p0 ** 3) * (2.0 ** (-m - 2 * h))

    # Symmetric inter-cluster contributions (approximated over the dominant LCA layer h)
    for h in range(1, H_int + 1):
        delta_N_h = n0 * (b ** (h - 1)) * (b - 1)
        pairs_in_shell = 0.5 * delta_N_h * (delta_N_h - n0 * (b ** (h - 1)))
        E_T_inter += pairs_in_shell * (p0 ** 3) * (8.0 ** (-h))

    return np.array([E_D, Var_D, E_T_intra + E_T_mixed + E_T_inter])


def invert_hsbm_parameters(g):
    """
    Solves the non-linear system of equations to invert parameters from a graph-tool object.
    """
    N = g.num_vertices()
    mean_d, var_d, mean_t = extract_graph_moments(g)

    # Objective function minimizing the delta between empirical and theoretical moments
    def equations(params):
        theoretical = theoretical_moments(params, N)
        return theoretical - np.array([mean_d, var_d, mean_t])

    # Initial heuristic guess [n0, b, p0]
    initial_guess = [32.0, 2.0, 0.5]

    # Perform multi-dimensional root finding using a hybrid Powell/Levenberg-Marquardt method
    result = root(equations, initial_guess, method='hybr')

    if result.success:
        n0_est, b_est, p0_est = result.x
        return {
            "n0": n0_est,
            "b": b_est,
            "p0": p0_est,
            "nc_guess": int(np.round(n0_est))  # Optimal target core size for hardware allocation
        }
    else:
        raise RuntimeError(f"Parameter inversion did not converge: {result.message}")


def run_estimator_validation():
    """
    Harness function to generate a known directed hSBM graph and validate the analytical estimators.
    """
    # 1. Define ground truth parameters
    true_n0 = 60.0
    true_b = 2.0
    true_H = 3  # Set to 3 for faster verification loops. Increase to 5 for larger networks.
    true_p0 = 0.4

    # 2. Generate the directed graph
    g = generate_verification_graph(n0=int(true_n0), b=int(true_b), H=true_H, p0=true_p0)

    print("\n==========================================")
    print("RUNNING DIRECTED ESTIMATOR INVERSION VALIDATION")
    print("==========================================")

    # 3. Extract traits and run the inversion solver
    try:
        estimated_params = invert_hsbm_parameters(g)

        print("\nGround Truth vs. Inverted Estimates:")
        print(f"Parameter | Ground Truth | Estimated Value")
        print(f"----------|--------------|----------------")
        print(f"n0        | {true_n0:<12} | {estimated_params['n0']:.4f}")
        print(f"b         | {true_b:<12} | {estimated_params['b']:.4f}")
        print(f"p0        | {true_p0:<12} | {estimated_params['p0']:.4f}")
        print(f"----------|--------------|----------------")
        print(f"Suggested Hardware Core Size (nc): {estimated_params['nc_guess']}")

    except Exception as e:
        print(f"\nInversion failed or did not converge: {str(e)}")


if __name__ == "__main__":
    run_estimator_validation()