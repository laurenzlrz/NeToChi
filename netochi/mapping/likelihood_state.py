import numpy as np
import graph_tool.all as gt
from netochi.mapping.hardware_config import HardwareConfig

from pydantic import BaseModel

class MappingResult(BaseModel):
    """Immutable results of a mapping execution."""
    total_edges: int
    valid_edges: int
    inconsistencies: int
    consistency_pct: float
    log_likelihood: float

class MappingState:
    def __init__(self, graph: gt.Graph, config: HardwareConfig, alpha: float = 0.9, epsilon: float = 0.1):
        self.g = graph
        self.config = config
        self.alpha = alpha
        self.epsilon = epsilon
        self.N = graph.num_vertices()
        self.m = graph.num_edges()
        
        # Degrees
        self.k_in = graph.get_in_degrees(graph.get_vertices())
        self.k_out = graph.get_out_degrees(graph.get_vertices())
        
        # State variables
        self.c = np.zeros(self.N, dtype=int)
        self.x = np.zeros(self.N, dtype=int)
        # slice assignment at distance d for node i: s[i, d]
        self.s = np.zeros((self.N, config.max_distance + 1), dtype=int)
        
        # Precompute lists of in/out edges for fast E_valid updates
        # graph-tool vertices iteration
        self.in_edges = [list(v.in_neighbors()) for v in graph.vertices()]
        self.out_edges = [list(v.out_neighbors()) for v in graph.vertices()]
        
        # Precompute cores at distance d
        # cores_at_dist[c][d] = list of cores at distance d from core c
        self.cores_at_dist = [[[] for _ in range(config.max_distance + 1)] for _ in range(config.total_cores)]
        for c1 in range(config.total_cores):
            for c2 in range(config.total_cores):
                d = config.core_distance(c1, c2)
                self.cores_at_dist[c1][d].append(c2)
                
    def init_random(self, seed=None):
        rng = np.random.default_rng(seed)
        
        # 1. Assign cores and local addresses
        # Create a list of all available slots (c, x)
        slots = [(c, x) for c in range(self.config.total_cores) for x in range(self.config.neurons_per_core)]
        rng.shuffle(slots)
        
        for i in range(self.N):
            self.c[i], self.x[i] = slots[i]
            
        # 2. Assign slices randomly
        for d in range(1, self.config.max_distance + 1):
            n_slices = self.config.num_slices_at_distance(d)
            self.s[:, d] = rng.integers(0, n_slices, size=self.N)

    def is_valid_edge(self, source: int, target: int) -> bool:
        """Check if edge from source to target is valid under current mapping."""
        c_src = self.c[source]
        c_tgt = self.c[target]
        dist = self.config.core_distance(c_tgt, c_src)
        if dist == 0:
            return True
        
        x_src = self.x[source]
        s_tgt = self.s[target, dist]
        start, end = self.config.get_slice_bounds(dist, s_tgt)
        return start <= x_src < end

    def compute_e_valid(self) -> int:
        """Compute the total number of valid edges."""
        e_valid = 0
        for tgt in range(self.N):
            for src in self.in_edges[tgt]:
                if self.is_valid_edge(int(src), tgt):
                    e_valid += 1
        return e_valid

    def compute_Z(self) -> float:
        """
        Compute normalization constant Z = sum_{u,v} k_u^{in} k_v^{out} W_{uv}
        W_{uv} = alpha * Delta_{uv} + epsilon * (1 - Delta_{uv})
        """
        # Calculate K = sum_{u,v} k_u^{in} k_v^{out} Delta_{uv}
        # To do this efficiently:
        # slice_out_mass[c, d, s] = sum_{v in core c, slice s at dist d} k_v^{out}
        
        slice_out_mass = np.zeros((self.config.total_cores, self.config.max_distance + 1, self.config.neurons_per_core))
        for v in range(self.N):
            c_v = self.c[v]
            x_v = self.x[v]
            k_v_out = self.k_out[v]
            
            # For dist 0, all nodes in core c are valid. We can store this in dist 0, slice 0.
            slice_out_mass[c_v, 0, 0] += k_v_out
            
            for d in range(1, self.config.max_distance + 1):
                # find which slice x_v belongs to at distance d
                n_slices = self.config.num_slices_at_distance(d)
                s_v = (x_v * n_slices) // self.config.neurons_per_core
                slice_out_mass[c_v, d, s_v] += k_v_out
                
        K = 0.0
        for u in range(self.N):
            c_u = self.c[u]
            k_u_in = self.k_in[u]
            
            # Dist 0
            mass_0 = slice_out_mass[c_u, 0, 0]
            K += k_u_in * mass_0
            
            # Dist > 0
            for d in range(1, self.config.max_distance + 1):
                s_u = self.s[u, d]
                mass_d = 0.0
                for c_src in self.cores_at_dist[c_u][d]:
                    mass_d += slice_out_mass[c_src, d, s_u]
                K += k_u_in * mass_d
                
        # Z = epsilon * m^2 + (alpha - epsilon) * K
        m2 = float(self.m) * float(self.m)
        Z = self.epsilon * m2 + (self.alpha - self.epsilon) * K
        return Z

    def log_likelihood(self) -> float:
        """Compute L(Theta) propto -m log Z + E_valid log(alpha) + E_invalid log(epsilon)"""
        if self.m == 0:
            return 0.0
        Z = self.compute_Z()
        e_valid = self.compute_e_valid()
        e_invalid = self.m - e_valid
        
        ll = -self.m * np.log(Z) + e_valid * np.log(self.alpha) + e_invalid * np.log(self.epsilon)
        return ll

    def inconsistencies(self) -> int:
        """
        Count the number of edges that violate hardware routing constraints.
        An inconsistent (invalid) edge is one where the source neuron's local
        address does not fall within the listening slice of the target neuron at
        the given routing distance. These are connections that cannot be physically
        realised on the hardware and would need to be dropped or rerouted.
        """
        return self.m - self.compute_e_valid()

    def mapping_stats(self) -> MappingResult:
        """
        Return a MappingResult with all key mapping quality metrics.
        """
        e_valid = self.compute_e_valid()
        e_invalid = self.m - e_valid
        pct = 100.0 * e_valid / self.m if self.m > 0 else 100.0
        Z = self.compute_Z()
        if self.m > 0:
            ll = -self.m * np.log(Z) + e_valid * np.log(self.alpha) + e_invalid * np.log(self.epsilon)
        else:
            ll = 0.0
        return MappingResult(
            total_edges=self.m,
            valid_edges=e_valid,
            inconsistencies=e_invalid,
            consistency_pct=pct,
            log_likelihood=ll,
        )
