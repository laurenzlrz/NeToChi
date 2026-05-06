import networkx as nx
import numpy as np

class WTAGenerator:
    """
    A Winner-Takes-All (WTA) network generator using a hub-and-spoke (Global Inhibition) skeleton.
    - Excitatory Pool: n-1 nodes.
    - Inhibitory Hub: 1 node (the last node).
    - Feedback: Excitatory -> Inhibitory with probability p.
    - Inhibition: Inhibitory -> Excitatory with probability p.
    """
    def __init__(self, n: int, p: float, seed: int | None = None):
        """
        Initialize the generator.
        
        Args:
            n: Total number of neurons (nodes). Must be at least 2.
            p: Connectivity probability for feedback and inhibition edges.
            seed: Random seed for reproducibility.
        """
        if n < 2:
            raise ValueError("WTA network requires at least 2 nodes (1 excitatory, 1 inhibitory).")
        self.n = n
        self.p = p
        self.seed = seed
        self._graph: nx.DiGraph | None = None

    def generate(self) -> nx.DiGraph:
        """
        Generate the directed WTA graph.
        
        Returns:
            The generated networkx.DiGraph.
        """
        rng = np.random.default_rng(self.seed)
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(range(self.n))
        
        excitatory_nodes = range(self.n - 1)
        inhibitory_hub = self.n - 1
        
        edges = []
        for e_node in excitatory_nodes:
            # 1. Feedback: Competitor to Referee
            if rng.random() < self.p:
                edges.append((e_node, inhibitory_hub))
            
            # 2. Inhibition: Referee to Competitor
            if rng.random() < self.p:
                edges.append((inhibitory_hub, e_node))
        
        self._graph.add_edges_from(edges)
        return self._graph

    @property
    def graph(self) -> nx.DiGraph | None:
        return self._graph
