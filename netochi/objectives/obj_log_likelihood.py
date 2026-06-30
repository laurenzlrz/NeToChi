import numpy as np
from typing import Dict, Any
from pydantic import BaseModel
import icontract

from netochi.mapping.interfaces import (
    BaseMosaicMappingState, 
)

from netochi.definitions.constants import LL_INVALID_PENALTY_LOG, LL_LAPLACIAN_SMOOTHING_NUM, \
    LL_LAPLACIAN_SMOOTHING_DEN, NAME_OBJ_LOG_LIKELIHOOD
from netochi.definitions.exceptions import BaselineMismatchError
from netochi.objectives.interfaces import ObjectiveInterface, MappingObjective, AbstractObjectiveConfig


class LogLikelihoodObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "LogLikelihoodObjective":
        return LogLikelihoodObjective(config=self)


class LogLikelihoodObjective(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]],
                             ObjectiveInterface[BaseMosaicMappingState[Any]]):
    """
    SBM-based Log-Likelihood objective for neuromorphic mapping.
    """

    @icontract.require(lambda config: isinstance(config, LogLikelihoodObjectiveConfig))
    def __init__(self, config: LogLikelihoodObjectiveConfig) -> None:
        self.config = config
        self._graph_cache: Dict[int, Any] = {}

    def evaluate(self, state: BaseMosaicMappingState[Any]) -> float:
        """Returns Negative Log-Likelihood (Energy)."""
        return -self.log_likelihood(state)

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[Any], baseline: BaseMosaicMappingState[Any]) -> float:
        """Evaluate relative difference in energy."""
        if id(state.mapping_input.graph) != id(baseline.mapping_input.graph):
            raise BaselineMismatchError("Graph topology mismatch between state and baseline.")
        return self.evaluate(state) - self.evaluate(baseline)

    def log_likelihood(self, state: BaseMosaicMappingState[Any]) -> float:
        """Calculate the log-likelihood for the current mapping assignment."""
        input_id = id(state.mapping_input)
        if input_id not in self._graph_cache:
            self._graph_cache[input_id] = self._precompute_graph(state)
            
        data = self._graph_cache[input_id]
        e_v = self._compute_e_valid(state, data)
        
        hw = state.hw_to_evaluate
        n0 = data['N'] * hw.neurons_per_core
        p0 = (e_v + LL_LAPLACIAN_SMOOTHING_NUM) / (n0 + LL_LAPLACIAN_SMOOTHING_DEN)
        
        ll = e_v * np.log(p0) + (data['m'] - e_v) * LL_INVALID_PENALTY_LOG
        return float(ll)

    def _precompute_graph(self, state: BaseMosaicMappingState[Any]) -> Dict[str, Any]:
        graph = state.mapping_input.graph
        return {
            'N': graph.num_vertices(),
            'm': graph.num_edges(),
            'in_edges': [[int(src) for src in v.in_neighbors()] for v in graph.vertices()]
        }

    def _compute_e_valid(self, state: BaseMosaicMappingState[Any], data: Dict[str, Any]) -> int:
        hw = state.hw_to_evaluate
        c, x, s = state.c, state.x, state.s
        e_valid = 0
        for tgt in range(data['N']):
            c_tgt = c[tgt]
            for src in data['in_edges'][tgt]:
                c_src = c[src]
                dist = hw.core_distance(int(c_tgt), int(c_src))
                if dist == 0:
                    e_valid += 1
                else:
                    if hw.get_slice_bounds(dist, s[tgt][dist])[0] <= x[src] < hw.get_slice_bounds(dist, s[tgt][dist])[1]:
                        e_valid += 1
        return e_valid

    def get_name(self) -> str:
        return NAME_OBJ_LOG_LIKELIHOOD
