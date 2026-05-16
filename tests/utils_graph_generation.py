
import graph_tool.all as gt


def create_fc_graph(n: int) -> gt.Graph:
    """Creates a single fully connected component of size n."""
    g = gt.Graph(directed=True)
    g.add_vertex(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                g.add_edge(i, j)
    return g


def create_two_fc_components(n: int) -> gt.Graph:
    """
    n: number neurons per component
    Creates two disconnected fully connected components of size n each.
    """
    g = gt.Graph(directed=True)
    g.add_vertex(2 * n)

    # Component A (Indices 0 to n-1)
    for i in range(n):
        for j in range(n):
            if i != j: g.add_edge(i, j)

    # Component B (Indices n to 2n-1)
    for i in range(n, 2 * n):
        for j in range(n, 2 * n):
            if i != j: g.add_edge(i, j)

    return g


def create_two_fc_with_directed_half(n: int) -> gt.Graph:
    """
    n: number neurons per component
    Creates two FC components (A and B).
    Exactly half of B connects to all neurons in A. No other inter-cluster edges.
    """
    assert n % 2 == 0, "n must be even to cleanly split half of cluster B."
    g = create_two_fc_components(n)

    # Half of B (Indices n to n + n//2 - 1) connects to all of A (Indices 0 to n-1)
    half_n = n // 2
    for b in range(n, n + half_n):
        g.add_edge(b, 0) # node ß receives input from half of b -> should have slice idx = 0

    return g
