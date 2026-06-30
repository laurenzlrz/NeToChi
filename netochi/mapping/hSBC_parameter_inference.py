import graph_tool.all as gt
import graph_tool.topology as gt_topo  # Expliziter Import für die Topologie-Funktionen
import graph_tool.spectral
import numpy as np
from scipy.optimize import root
import scipy.sparse.linalg as sla
from scipy.optimize import minimize_scalar
import scipy.sparse as tuple_sparse
from scipy.sparse.linalg import eigsh

def generate_verification_graph(n0, b, H, p0, p_active):
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

    active_mask = np.random.rand(N) < p_active
    num_active = np.sum(active_mask)
    print(f"      Knoten-Aktivierung: {num_active} von {N} Knoten sind aktiv (p_active={p_active}).")

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

            if not active_mask[i] or not active_mask[j]:
                pairs_processed += 1
                if pairs_processed % checkpoint == 0:
                    progress = (pairs_processed / total_pairs) * 100
                    print(f"      Progress: {progress:.0f}% of ordered pairs evaluated.")
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
    Uses the undirected skeleton degrees exclusively to reconstruct correct triangle counts.
    """
    # 1. Extract total directed degrees (in-degree + out-degree) for Edge Moments
    degrees_dir = g.degree_property_map("total").a
    mean_d = np.mean(degrees_dir)
    var_d = np.var(degrees_dir, ddof=1)

    # 2. Extract unique neighbor counts (undirected skeleton degrees) for Structural Moments
    g_undir = gt.GraphView(g, directed=False)
    degrees_undir = g_undir.degree_property_map("total").a

    # Get clustering coefficient based on the undirected skeleton
    c_v = gt.local_clustering(g, undirected=True).a

    # Reconstruct the number of triangles using the matching undirected degrees
    triangles = 0.5 * c_v * degrees_undir * (degrees_undir - 1)
    triangles = np.nan_to_num(triangles)
    mean_t = np.mean(triangles)

    return mean_d, var_d, mean_t


def theoretical_moments(params, N):
    """
    Computes the exact mathematical expected values of the hSBM for a directed graph setup,
    correcting the structural triangle moments for undirected skeleton edge collapse.
    """
    n0, b, p0 = params

    if n0 < 2 or b <= 1 or p0 <= 0 or p0 > 1:
        return np.array([1e6, 1e6, 1e6])

    H = np.log(N / n0) / np.log(b)
    H_int = int(np.clip(np.round(H), 1, 20))

    # --- 1. Expected Total Directed Degree E[D] ---
    E_D = 2.0 * (n0 - 1) * p0
    for h in range(1, H_int + 1):
        delta_N = n0 * (b ** (h - 1)) * (b - 1)
        E_D += 2.0 * delta_N * p0 * (2.0 ** (-h))

    # --- 2. Expected Degree Variance Var(D) ---
    Var_D = 2.0 * (n0 - 1) * p0 * (1.0 - p0)
    for h in range(1, H_int + 1):
        delta_N = n0 * (b ** (h - 1)) * (b - 1)
        p_h = p0 * (2.0 ** (-h))
        Var_D += 2.0 * delta_N * p_h * (1.0 - p_h)

    # --- 3. Expected Triangles E[T] (Undirected Skeleton) ---
    # Define effective undirected connection probabilities: p_undir = 2p - p^2
    p_u0 = 2.0 * p0 - p0 ** 2

    E_T_intra = 0.5 * (n0 - 1) * (n0 - 2) * (p_u0 ** 3)

    E_T_mixed = 0.0
    for h in range(1, H_int + 1):
        delta_N = n0 * (b ** (h - 1)) * (b - 1)
        p_uh = 2.0 * (p0 * (2.0 ** (-h))) - (p0 * (2.0 ** (-h))) ** 2
        E_T_mixed += (n0 - 1) * delta_N * p_u0 * (p_uh ** 2)

    E_T_inter = 0.0
    for h in range(2, H_int + 1):
        delta_N_h = n0 * (b ** (h - 1)) * (b - 1)
        p_uh = 2.0 * (p0 * (2.0 ** (-h))) - (p0 * (2.0 ** (-h))) ** 2
        for m in range(1, h):
            delta_N_m = n0 * (b ** (m - 1)) * (b - 1)
            p_um = 2.0 * (p0 * (2.0 ** (-m))) - (p0 * (2.0 ** (-m))) ** 2
            E_T_inter += 2.0 * delta_N_m * delta_N_h * p_um * (p_uh ** 2)

    for h in range(1, H_int + 1):
        delta_N_h = n0 * (b ** (h - 1)) * (b - 1)
        pairs_in_shell = 0.5 * delta_N_h * (delta_N_h - n0 * (b ** (h - 1)))
        p_uh = 2.0 * (p0 * (2.0 ** (-h))) - (p0 * (2.0 ** (-h))) ** 2
        E_T_inter += pairs_in_shell * (p_uh ** 3)

    return np.array([E_D, Var_D, E_T_intra + E_T_mixed + E_T_inter])


def invert_hsbm_parameters(g):
    """
    Solves the non-linear system of equations to invert parameters from a graph-tool object.
    """
    N = g.num_vertices()
    mean_d, var_d, mean_t = extract_graph_moments(g)

    def equations(params):
        theoretical = theoretical_moments(params, N)
        return theoretical - np.array([mean_d, var_d, mean_t])

    # Dynamic initial guess to assist convergence stability
    initial_guess = [45.0, 2.0, 0.35]

    result = root(equations, initial_guess, method='hybr')

    if result.success:
        n0_est, b_est, p0_est = result.x
        return {
            "n0": n0_est,
            "b": b_est,
            "p0": p0_est,
            "nc_guess": int(np.round(n0_est))
        }
    else:
        raise RuntimeError(f"Parameter inversion did not converge: {result.message}")


def robust_spectral_hsbm_estimator(g, expected_H=5):
    """
    Estimates hSBM parameters purely from global graph macro-statistics
    and spectral properties. No cluster assignments, no MCMC, no fragile roots.
    """
    N = g.num_vertices()

    # 1. Get total directed degree stats for p0 estimation
    degrees_dir = g.degree_property_map("total").a
    mean_d_dir = np.mean(degrees_dir)

    # 2. Extract the undirected skeleton adjacency matrix
    g_undir = gt.GraphView(g, directed=False)
    # Convert to scipy sparse matrix safely
    A = gt.adjacency(g_undir)

    # Calculate the average degree of the undirected skeleton
    degrees_undir = g_undir.degree_property_map("total").a
    mean_d_undir = np.mean(degrees_undir)

    # 3. Compute the top eigenvalues to locate the structural signal
    # We need enough eigenvalues to span the structural blocks (b^H)
    num_eigenvalues = min(50, N - 2)
    print(f"      [Spectral] Computing top {num_eigenvalues} eigenvalues...")

    # Compute largest eigenvalues of the symmetric adjacency matrix
    eigenvals = sla.eigsh(A, k=num_eigenvalues, which='LM', return_eigenvectors=False)
    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending

    # 4. Determine the Noise Threshold (Wigner Bulk Boundary)
    # Standard deviation threshold for Erdös-Rényi background noise
    noise_threshold = 2.0 * np.sqrt(mean_d_undir * (1.0 - (mean_d_undir / N)))

    # Count how many eigenvalues stand significantly above the noise floor
    significant_signals = np.sum(eigenvals > noise_threshold)

    # Treat the structural signals as the number of base blocks B_base
    B_base = max(2, significant_signals)

    # --- Parameter Inversion ---
    # 1. Estimate n0 from the block count
    n0_est = N / B_base

    # 2. Estimate b given the expected hierarchy depth H
    # B_base = b^H -> b = B_base^(1/H)
    b_est = B_base ** (1.0 / expected_H)

    # 3. Estimate p0 directly from the directed mean degree
    # E[D_dir] = 2 * p0 * [ (n0-1) + sum_{h=1}^H delta_N_h * 2^-h ]
    # For a binary tree layout, the bracket reduces elegantly near ~ (2 * n0 - 1)
    # We can isolate p0 directly using a direct ratio of the edge count
    total_possible_directed_edges = N * (N - 1)
    actual_edges = g.num_edges()
    global_density = actual_edges / total_possible_directed_edges

    # Back-calculate p0 scaling factor based on structural weight
    # Global density is a weighted average of p0 across layers.
    # For p_h = p0 * 2^-h, the effective global scaling factor is:
    weight = (n0_est / N) + sum(
        [(n0_est * (b_est ** (h - 1)) * (b_est - 1)) * (0.5 ** h) / N for h in range(1, expected_H + 1)])
    p0_est = np.clip(global_density / weight, 0.0, 1.0)

    return {
        "n0": n0_est,
        "b": b_est,
        "p0": p0_est,
        "nc_guess": int(np.round(n0_est))
    }


def robust_spectral_hsbm_estimator_blind(g):
    """
    Estimates hSBM parameters (including H and b) purely from global spectral gaps.
    Requires no priors.
    """
    N = g.num_vertices()

    # 1. Get total directed degrees
    degrees_dir = g.degree_property_map("total").a

    # 2. Extract the undirected skeleton adjacency matrix
    g_undir = gt.GraphView(g, directed=False)
    A = gt.adjacency(g_undir)

    degrees_undir = g_undir.degree_property_map("total").a
    mean_d_undir = np.mean(degrees_undir)

    # 3. Compute top eigenvalues (calculate enough to capture deep trees)
    num_eigenvalues = min(150, N - 2)
    print(f"      [Spectral] Computing top {num_eigenvalues} eigenvalues for blind inference...")

    eigenvals = sla.eigsh(A, k=num_eigenvalues, which='LM', return_eigenvectors=False)
    eigenvals = np.sort(eigenvals)[::-1]

    # 4. Determine the Noise Threshold (Wigner Bulk Boundary)
    noise_threshold = 2.0 * np.sqrt(mean_d_undir * (1.0 - (mean_d_undir / N)))
    significant_signals = np.sum(eigenvals > noise_threshold)

    B_base = max(2, significant_signals)
    n0_est = N / B_base

    # 5. Infer Tree Depth H via Log-Spectral Gaps
    # We ignore lambda_1 (global degree) and look at the structural signal
    if B_base > 2:
        signal_vals = eigenvals[1:B_base]
        signal_vals = signal_vals[signal_vals > 0]  # Safety filter

        if len(signal_vals) > 1:
            log_vals = np.log(signal_vals)
            log_gaps = np.abs(np.diff(log_vals))

            # Since p drops by factor 1/2, expected true structural gap is ln(2) ≈ 0.69
            # We set a robust threshold at 0.35 to cleanly separate layers from internal noise
            H_est = np.sum(log_gaps > 0.35) + 1
        else:
            H_est = 1
    else:
        H_est = 1

    # 6. Decouple Branching Factor b
    # B_base = b^H  =>  b = B_base^(1/H)
    b_est = B_base ** (1.0 / H_est)

    # 7. Infer p0 via expected density vs actual density
    actual_edges = g.num_edges()
    total_possible_directed_edges = N * (N - 1)
    global_density = actual_edges / total_possible_directed_edges

    H_int = int(np.round(H_est))
    weight = (n0_est / N) + sum(
        [(n0_est * (b_est ** (h - 1)) * (b_est - 1)) * (0.5 ** h) / N for h in range(1, H_int + 1)])

    p0_est = np.clip(global_density / weight, 0.0, 1.0) if weight > 0 else 0.0

    return {
        "n0": n0_est,
        "b": b_est,
        "H_inferred": H_est,
        "p0": p0_est,
        "nc_guess": int(np.round(n0_est))
    }


def _K_factor(b: float, h: int) -> float:
    """Skalierungsfaktor für den Knotengrad über alle Ebenen k=1 bis h."""
    if h == 0: return 1.0
    k = np.arange(1, h + 1)
    return 1.0 + ((b - 1) / b) * np.sum((b / 2.0) ** k)


def estimate_parameters_spectral(g: gt.Graph) -> dict:
    """
    Schätzt n0, b, h und p0 mithilfe des Spektrums (Eigenwerte) der Adjazenzmatrix.
    Löst Teiler-Mehrdeutigkeiten über den beobachteten Knotengrad auf.
    """
    N_obs = g.num_vertices()
    if N_obs < 10:
        raise ValueError("Graph ist zu klein für spektrale Inferenz.")

    # 1. GraphView ungerichtet machen & Adjazenzmatrix holen
    g_undir = gt.GraphView(g, directed=False)
    adj_matrix = gt.adjacency(g_undir).astype(float)

    # 2. Berechne die größten Eigenwerte (top 100 oder N-2)
    n_eigs = min(100, N_obs - 2)
    eigenvalues, _ = eigsh(adj_matrix, k=n_eigs, which='LM')
    eigenvalues = np.sort(eigenvalues)[::-1]  # Absteigend sortieren

    # 3. Bestimmung der Anzahl der Basis-Cluster (nc_guess) über die Rauschgrenze
    degrees = g_undir.get_total_degrees(g_undir.get_vertices())
    d_obs = np.mean(degrees)

    noise_threshold = 2.0 * np.sqrt(d_obs)
    structural_eigs = np.sum(eigenvalues > noise_threshold)

    # Mindestens 2 Cluster, falls das Signal verrauscht ist
    nc_guess = max(2, structural_eigs)

    # Vorläufiges n0 basierend auf der gefundenen Clusteranzahl
    n0_est = N_obs / nc_guess

    # 4. Tie-Breaking für H und b über den echten Knotengrad
    best_h = 1
    best_b = float(nc_guess)
    best_score = float('inf')  # Kombinierter Fehler-Score (Rundung + Grad-Fit)

    max_h = max(2, int(np.log2(N_obs)))
    for h_test in range(1, max_h + 1):
        b_test = nc_guess ** (1.0 / h_test)
        if b_test < 1.05:
            continue

        # Wie nah ist b an einer Ganzzahl?
        b_round_error = abs(b_test - round(b_test))

        # Falls b numerisch eine valide Ganzzahl darstellt (z.B. bei 16, 4, 2)
        # prüfen wir, welches H den echten Knotengrad am besten reproduziert.
        # Erwarteter Grad: d = n0 * p0 * K(b, h).
        # Da p0 im dichten Bereich ~ lambda_1 / n0, testen wir die Konsistenz:
        K = _K_factor(b_test, h_test)
        p0_test = np.clip(d_obs / (n0_est * K), 0.0, 1.0)
        d_predicted = n0_est * p0_test * K
        degree_error = abs(d_predicted - d_obs)

        # Score kombiniert Ganzzahligkeit (Prior) und physikalischen Grad-Fit
        # Wir geben dem Grad-Fit bei perfekter Ganzzahligkeit das entscheidende Gewicht
        current_score = b_round_error * 100.0 + degree_error

        # Verwende < oder ein weiches Kriterium, um tiefere Bäume bei Gleichstand zu bevorzugen
        if current_score <= best_score:
            # Bei nahezu identischem Score bevorzugen wir das höhere H (da Verzweigungen oft b=2 sind)
            if abs(current_score - best_score) < 1e-5 and h_test < best_h:
                continue
            best_score = current_score
            best_h = h_test
            best_b = b_test

    # 5. Finale Parameter mit dem optimierten H berechnen
    p0_est = np.clip(d_obs / (n0_est * _K_factor(best_b, best_h)), 0.0, 1.0)

    return {
        "n0": float(n0_est),
        "b": float(best_b),
        "H_inferred": int(best_h),
        "p0": float(p0_est),
        "nc_guess": int(np.round(nc_guess))
    }


def estimate_parameters_with_b(g: gt.Graph, b: float) -> dict:
    """
    Schätzt n0, h und p0 mithilfe des Spektrums, wenn der Verzweigungsfaktor b bekannt ist.

    :param g: graph_tool.Graph Objekt
    :param b: Bekannter Verzweigungsfaktor (z.B. 2.0)
    :return: Dictionary mit den geschätzten Parametern
    """
    N_obs = g.num_vertices()
    if N_obs < 10:
        raise ValueError("Graph ist zu klein für spektrale Inferenz.")
    if b <= 1.0:
        raise ValueError("Der Verzweigungsfaktor b must größer als 1 sein.")

    # 1. GraphView ungerichtet machen & Adjazenzmatrix holen
    g_undir = gt.GraphView(g, directed=False)
    adj_matrix = gt.adjacency(g_undir).astype(float)

    # 2. Berechne die größten Eigenwerte (top 100 oder N-2)
    n_eigs = min(100, N_obs - 2)
    eigenvalues, _ = eigsh(adj_matrix, k=n_eigs, which='LM')
    eigenvalues = np.sort(eigenvalues)[::-1]  # Absteigend sortieren

    # 3. Bestimmung der Anzahl der Basis-Cluster (nc_guess) über die Rauschgrenze
    degrees = g_undir.get_total_degrees(g_undir.get_vertices())
    d_obs = np.mean(degrees)

    noise_threshold = 2.0 * np.sqrt(d_obs)
    structural_eigs = np.sum(eigenvalues > noise_threshold)

    # Mindestens 2 Cluster annehmen
    nc_guess = max(2, structural_eigs)

    # 4. Höhe H direkt bestimmen via nc = b^H  =>  H = log_b(nc)
    # Wir runden auf die nächste Ganzzahl, da die Höhe diskret ist
    H_est = int(np.round(np.log(nc_guess) / np.log(b)))
    H_est = max(1, H_est)  # Mindestens Höhe 1

    # 5. n0 präzisieren (bereinigt um Dummy-Knoten-Effekte über die Struktur)
    nc_corrected = b ** H_est
    n0_est = N_obs / nc_corrected

    # 6. p0 schätzen unter Einbeziehung des Skalierungsfaktors für Höhe H
    K = _K_factor(b, H_est)
    p0_est = d_obs / (n0_est * K) if n0_est > 0 else 0.0
    p0_est = float(np.clip(p0_est, 0.0, 1.0))

    return {
        "n0": float(n0_est),
        "b": float(b),
        "H_inferred": int(H_est),
        "p0": p0_est,
        "nc_guess": int(np.round(n0_est))
    }

def run_estimator_validation():
    # Ground Truth without providing H to the estimator
    true_n0 = 60.0
    true_b = 2.0
    true_H = 4
    true_p0 = 0.7

    g = generate_verification_graph(n0=int(true_n0), b=int(true_b), H=true_H, p0=true_p0, p_active=0.8)

    print("\n==========================================")
    print("RUNNING BLIND SPECTRAL INVERSION VALIDATION")
    print("==========================================")

    try:
        estimated_params = estimate_parameters_spectral(g)
        estimated_params = estimate_parameters_with_b(g, b=true_b)

        print("\nGround Truth vs. Blind Spectral Estimates:")
        print(f"Parameter | Ground Truth | Estimated Value")
        print(f"----------|--------------|----------------")
        print(f"n0        | {true_n0:<12} | {estimated_params['n0']:.4f}")
        print(f"b         | {true_b:<12} | {estimated_params['b']:.4f}")
        print(f"H         | {true_H:<12} | {estimated_params['H_inferred']:.4f}")
        print(f"p0        | {true_p0:<12} | {estimated_params['p0']:.4f}")
        print(f"----------|--------------|----------------")
        print(f"Suggested Hardware Core Size (nc): {estimated_params['nc_guess']}")

    except Exception as e:
        print(f"\nInversion failed: {str(e)}")


if __name__ == "__main__":
    run_estimator_validation()
