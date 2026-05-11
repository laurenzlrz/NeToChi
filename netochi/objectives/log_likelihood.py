from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from netochi.objectives.interfaces import MappingObjective
from netochi.mapping.interfaces import MosaicMappingState
from netochi.input_generator.interfaces import MosaicMappingInput

class PrecomputedConnectivityData(BaseModel):
    """Container for static connectivity data used in likelihood calculations."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    N: int
    m: int
    k_in: np.ndarray
    k_out: np.ndarray
    in_edges: List[List[int]]
    cores_at_dist: List[List[List[int]]]

class LogLikelihoodObjectiveInterface(ABC):
    """Specific interface for Log-Likelihood based objectives."""
    
    @abstractmethod
    def log_likelihood(self, state: MosaicMappingState) -> float:
        """Returns Log-Likelihood for the given state."""
        pass

    @abstractmethod
    def precompute(self, mapping_input: MosaicMappingInput) -> PrecomputedConnectivityData:
        """Precompute static connectivity data once per input."""
        pass

class LogLikelihoodObjective(MappingObjective[MosaicMappingState], LogLikelihoodObjectiveInterface):
    """
    Section 5 Stochastic Block Model (SBM) Log-Likelihood objective.
    
    Truly decoupled evaluator. Precomputes and caches connectivity data 
    on the fly when a new MappingInput is encountered.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alpha: float = Field(default=0.9, description="Likelihood of compliant edges.")
    epsilon: float = Field(default=0.1, description="Noise probability for non-compliant edges.")
    
    # Internal cache for precomputed data: mapping_input_id -> PrecomputedConnectivityData
    _cache: Dict[int, PrecomputedConnectivityData] = PrivateAttr(default_factory=dict)

    def evaluate(self, state: MosaicMappingState) -> float:
        """Returns -LogLikelihood for the given state."""
        return -self.log_likelihood(state)

    def log_likelihood(self, state: MosaicMappingState) -> float:
        """Section 5 Log-Likelihood calculation."""
        mapping_input = state.mapping_input
        input_id = id(mapping_input)
        
        if input_id not in self._cache:
            self._cache[input_id] = self.precompute(mapping_input)
            
        data = self._cache[input_id]
        
        c = state.c
        x = state.x
        s = state.s
        
        if data.m == 0: 
            return 0.0

        Z = self._compute_Z(state, data)
        e_v = self._compute_e_valid(state, data)
        e_i = data.m - e_v
        
        return -data.m * np.log(Z) + e_v * np.log(self.alpha) + e_i * np.log(self.epsilon)

    def precompute(self, mapping_input: MosaicMappingInput) -> PrecomputedConnectivityData:
        """Precompute static connectivity data once per input."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        
        N = graph.num_vertices()
        total_cores = hw.total_cores
        max_dist = hw.max_distance
        
        k_in = graph.get_in_degrees(graph.get_vertices())
        k_out = graph.get_out_degrees(graph.get_vertices())
        
        in_edges = [[int(src) for src in v.in_neighbors()] for v in graph.vertices()]
        
        cores_at_dist = [[[] for _ in range(max_dist + 1)] for _ in range(total_cores)]
        for c1 in range(total_cores):
            for c2 in range(total_cores):
                d = hw.core_distance(c1, c2)
                cores_at_dist[c1][d].append(c2)
                
        return PrecomputedConnectivityData(
            N=N,
            m=graph.num_edges(),
            k_in=k_in,
            k_out=k_out,
            in_edges=in_edges,
            cores_at_dist=cores_at_dist
        )

    def _compute_e_valid(self, state: MosaicMappingState, data: PrecomputedConnectivityData) -> int:
        """Count edges satisfying Fan-In constraints."""
        hw = state.mapping_input.hw_config
        c = state.c
        x = state.x
        s = state.s
        
        e_valid = 0
        for tgt in range(data.N):
            c_tgt = c[tgt]
            for src in data.in_edges[tgt]:
                c_src = c[src]
                dist = hw.core_distance(c_tgt, c_src)
                if dist == 0:
                    e_valid += 1
                else:
                    x_src = x[src]
                    s_tgt_d = s[tgt, dist]
                    start, end = hw.get_slice_bounds(dist, s_tgt_d)
                    if start <= x_src < end:
                        e_valid += 1
        return e_valid

    def _compute_Z(self, state: MosaicMappingState, data: PrecomputedConnectivityData) -> float:
        """Compute normalization constant Z using O(N) slice-summation."""
        hw = state.mapping_input.hw_config
        c = state.c
        x = state.x
        s = state.s
        
        total_cores = hw.total_cores
        max_dist = hw.max_distance
        neurons_per_core = hw.neurons_per_core
        
        slice_out_mass = np.zeros((total_cores, max_dist + 1, neurons_per_core))
        
        for v in range(data.N):
            c_v = c[v]
            x_v = x[v]
            k_v_out = data.k_out[v]
            
            slice_out_mass[c_v, 0, 0] += k_v_out
            for d in range(1, max_dist + 1):
                n_sl = hw.num_slices_at_distance(d)
                s_v = (x_v * n_sl) // neurons_per_core
                slice_out_mass[c_v, d, s_v] += k_v_out
                
        K = 0.0
        for u in range(data.N):
            c_u = c[u]
            k_u_in = data.k_in[u]
            
            K += k_u_in * slice_out_mass[c_u, 0, 0]
            
            for d in range(1, max_dist + 1):
                s_u_d = s[u, d]
                mass_d = 0.0
                for c_src in data.cores_at_dist[c_u][d]:
                    mass_d += slice_out_mass[c_src, d, s_u_d]
                K += k_u_in * mass_d
                
        Z = self.epsilon * (float(data.m)**2) + (self.alpha - self.epsilon) * K
        return Z
