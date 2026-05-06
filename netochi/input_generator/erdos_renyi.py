import networkx as nx

class ErdosRenyiGenerator:
    """
    A simple Erdős-Rényi network generator.
    Generates a G(n, p) directed graph where each possible edge exists with probability p.
    """
    def __init__(self, n: int, p: float, seed: int | None = None):
        """
        Initialize the generator.
        
        Args:
            n: Number of neurons (nodes).
            p: Probability for edge creation.
            seed: Random seed for reproducibility.
        """
        self.n = n
        self.p = p
        self.seed = seed
        self._graph: nx.DiGraph | None = None

    def generate(self) -> nx.DiGraph:
        """
        Generate the directed Erdős-Rényi graph.
        
        Returns:
            The generated networkx.DiGraph.
        """
        self._graph = nx.fast_gnp_random_graph(self.n, self.p, seed=self.seed, directed=True)
        return self._graph

    @property
    def graph(self) -> nx.DiGraph | None:
        return self._graph
