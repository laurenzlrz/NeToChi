"""
Likelihood calculation for the Section 5 Stochastic Block Model (SBM).

Refactored to follow the "Großprojekt" Pydantic standard.
Provides the MappingState which calculates Log-Likelihood and stats.
"""

import numpy as np
import graph_tool.all as gt
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field
from typing import List, Any, Optional

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


class MappingResult(BaseModel):
    """Immutable results of a mapping execution."""
    model_config = ConfigDict(frozen=True)

    total_edges: int
    valid_edges: int
    inconsistencies: int
    consistency_pct: float
    log_likelihood: float


class MappingState(BaseModel):
    """
    Core mathematical state representing a mapping and its likelihood.
    
    Uses Pydantic for configuration and rohe NumPy-Arrays via PrivateAttr
    for high-performance calculations in the MCMC loop.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration and Inputs
    graph: Any = Field(description="The graph-tool Graph object.")
    config: MosaicHardwareConfig = Field(description="Hardware configuration.")
    alpha: float = Field(default=0.9, description="Likelihood of compliant edges.")
    epsilon: float = Field(default=0.1, description="Noise probability for non-compliant edges.")

    # Private high-performance attributes
    _c: np.ndarray = PrivateAttr()  # core_idx assignment
    _x: np.ndarray = PrivateAttr()  # local_idx assignment
    _s: np.ndarray = PrivateAttr()  # slice assignment
    
    _N: int = PrivateAttr()
    _m: int = PrivateAttr()
    _k_in: np.ndarray = PrivateAttr()
    _k_out: np.ndarray = PrivateAttr()
    
    _in_edges: List[List[int]] = PrivateAttr()
    _out_edges: List[List[int]] = PrivateAttr()
    _cores_at_dist: List[List[List[int]]] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        
        # Extract graph info
        self._N = self.graph.num_vertices()
        self._m = self.graph.num_edges()
        
        # Degrees
        self._k_in = self.graph.get_in_degrees(self.graph.get_vertices())
        self._k_out = self.graph.get_out_degrees(self.graph.get_vertices())
        
        # Initialize assignment arrays
        self._c = np.zeros(self._N, dtype=int)
        self._x = np.zeros(self._N, dtype=int)
        self._s = np.zeros((self._N, self.config.max_distance + 1), dtype=int)
        
        # Precompute edge lists for faster validation
        # (graph-tool iterators can be slow in loops)
        self._in_edges = [ [int(src) for src in v.in_neighbors()] for v in self.graph.vertices() ]
        self._out_edges = [ [int(tgt) for tgt in v.out_neighbors()] for v in self.graph.vertices() ]
        
        # Precompute distance-based core lookups
        max_dist = self.config.max_distance
        total_cores = self.config.total_cores
        self._cores_at_dist = [[[] for _ in range(max_dist + 1)] for _ in range(total_cores)]
        for c1 in range(total_cores):
            for c2 in range(total_cores):
                d = self.config.core_distance(c1, c2)
                self._cores_at_dist[c1][d].append(c2)

    # --- Properties for public access (used by Mappers) ---

    @property
    def c(self) -> np.ndarray: return self._c
    
    @c.setter
    def c(self, value: np.ndarray): self._c = value

    @property
    def x(self) -> np.ndarray: return self._x
    
    @x.setter
    def x(self, value: np.ndarray): self._x = value

    @property
    def s(self) -> np.ndarray: return self._s
    
    @s.setter
    def s(self, value: np.ndarray): self._s = value

    @property
    def N(self) -> int: return self._N

    @property
    def m(self) -> int: return self._m

    # --- Logic ---

    def init_random(self, seed: Optional[int] = None):
        """Randomly initialize assignments respecting hardware capacities."""
        rng = np.random.default_rng(seed)
        
        # 1. Distribute across cores evenly
        slots = [(c, x) for c in range(self.config.total_cores) for x in range(self.config.neurons_per_core)]
        rng.shuffle(slots)
        
        for i in range(self._N):
            self._c[i], self._x[i] = slots[i]
            
        # 2. Randomize slices
        for d in range(1, self.config.max_distance + 1):
            n_sl = self.config.num_slices_at_distance(d)
            self._s[:, d] = rng.integers(0, n_sl, size=self._N)

    def is_valid_edge(self, source: int, target: int) -> bool:
        """Check if edge satisfies hardware Fan-In constraints."""
        c_src = self._c[source]
        c_tgt = self._c[target]
        dist = self.config.core_distance(c_tgt, c_src)
        if dist == 0: return True
        
        x_src = self._x[source]
        s_tgt = self._s[target, dist]
        start, end = self.config.get_slice_bounds(dist, s_tgt)
        return start <= x_src < end

    def compute_e_valid(self) -> int:
        """Total number of valid edges under current mapping."""
        e_valid = 0
        for tgt in range(self._N):
            for src in self._in_edges[tgt]:
                if self.is_valid_edge(src, tgt):
                    e_valid += 1
        return e_valid

    def compute_Z(self) -> float:
        """
        Compute normalization constant Z using the O(N) slice-summation trick.
        Z = sum_{u,v} k_u_in * k_v_out * W_uv
        """
        total_cores = self.config.total_cores
        max_dist = self.config.max_distance
        neurons_per_core = self.config.neurons_per_core
        
        # slice_out_mass[c, d, s] = sum of k_out for all nodes in core c in slice s (at dist d)
        slice_out_mass = np.zeros((total_cores, max_dist + 1, neurons_per_core))
        
        for v in range(self._N):
            c_v = self._c[v]
            x_v = self._x[v]
            k_v_out = self._k_out[v]
            
            slice_out_mass[c_v, 0, 0] += k_v_out
            for d in range(1, max_dist + 1):
                n_sl = self.config.num_slices_at_distance(d)
                s_v = (x_v * n_sl) // neurons_per_core
                slice_out_mass[c_v, d, s_v] += k_v_out
                
        K = 0.0
        for u in range(self._N):
            c_u = self._c[u]
            k_u_in = self._k_in[u]
            
            # Distance 0 mass
            K += k_u_in * slice_out_mass[c_u, 0, 0]
            
            # Hierarchical distance mass
            for d in range(1, max_dist + 1):
                s_u = self._s[u, d]
                mass_d = 0.0
                for c_src in self._cores_at_dist[c_u][d]:
                    mass_d += slice_out_mass[c_src, d, s_u]
                K += k_u_in * mass_d
                
        # Z = epsilon * m^2 + (alpha - epsilon) * K
        m_f = float(self._m)
        Z = self.epsilon * (m_f * m_f) + (self.alpha - self.epsilon) * K
        return Z

    def log_likelihood(self) -> float:
        """Section 5 Log-Likelihood."""
        if self._m == 0: return 0.0
        Z = self.compute_Z()
        e_v = self.compute_e_valid()
        e_i = self._m - e_v
        
        return -self._m * np.log(Z) + e_v * np.log(self.alpha) + e_i * np.log(self.epsilon)

    def mapping_stats(self) -> MappingResult:
        """Return quality metrics for the current mapping."""
        e_v = self.compute_e_valid()
        e_i = self._m - e_v
        pct = 100.0 * e_v / self._m if self._m > 0 else 100.0
        ll = self.log_likelihood()
        
        return MappingResult(
            total_edges=self._m,
            valid_edges=e_v,
            inconsistencies=e_i,
            consistency_pct=pct,
            log_likelihood=ll,
        )
