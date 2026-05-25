from typing import Dict, Any, Generic
from pydantic import PrivateAttr, ConfigDict

from netochi.objectives.interfaces import NetworkMappingObjective
from netochi.mapping.interfaces import (
    BaseMosaicMappingState, 
    NetworkAssignmentState,
    ANY_MAPPING_INPUT
)
from netochi.input_generator.generics import WITH_HW_INPUT
from netochi.objectives.exceptions import BaselineMismatchError


class InconsistencyObjective(NetworkMappingObjective[BaseMosaicMappingState[ANY_MAPPING_INPUT], BaseMosaicMappingState[ANY_MAPPING_INPUT]], Generic[ANY_MAPPING_INPUT, WITH_HW_INPUT]):
    """
    Objective that counts the number of invalid edges (edges violating hardware constraints).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    _graph_cache: Dict[int, Any] = PrivateAttr(default_factory=dict)

    def evaluate(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> float:
        """Returns total number of invalid edges."""
        input_id = id(state.mapping_input)
        if input_id not in self._graph_cache:
            self._graph_cache[input_id] = self._precompute_graph(state)
            
        data = self._graph_cache[input_id]
        e_v = self._compute_e_valid(state, data)
        return float(data['m'] - e_v)

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT], baseline: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> float:
        """Evaluate relative difference in inconsistencies."""
        if id(state.mapping_input.graph) != id(baseline.mapping_input.graph):
            raise BaselineMismatchError("Graph topology mismatch between state and baseline.")
        return self.evaluate(state) - self.evaluate(baseline)

    def _precompute_graph(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> Dict[str, Any]:
        graph = state.mapping_input.graph
        return {
            'N': graph.num_vertices(),
            'm': graph.num_edges(),
            'in_edges': [[int(src) for src in v.in_neighbors()] for v in graph.vertices()]
        }

    def _compute_e_valid(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT], data: Dict[str, Any]) -> int:
        hw = state.hw
        c, x, s = state.c, state.x, state.s
        e_valid = 0
        for tgt in range(data['N']):
            c_tgt = c[tgt]
            for src in data['in_edges'][tgt]:
                c_src = c[src]
                dist = hw.core_distance(c_tgt, c_src)
                if dist == 0:
                    e_valid += 1
                else:
                    if hw.get_slice_bounds(dist, s[tgt][dist])[0] <= x[src] < hw.get_slice_bounds(dist, s[tgt][dist])[1]:
                        e_valid += 1
        return e_valid
