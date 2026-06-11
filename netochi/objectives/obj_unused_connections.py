from typing import Generic, Any, Dict

from pydantic import PrivateAttr, ConfigDict

from netochi.objectives.interfaces import MappingObjective
from netochi.mapping.interfaces import BaseMosaicMappingState, ANY_MAPPING_INPUT
from netochi.objectives.utils import compute_e_valid, compute_total_hw_connections


class UnusedConnectionsObjective(MappingObjective[BaseMosaicMappingState[ANY_MAPPING_INPUT], BaseMosaicMappingState[ANY_MAPPING_INPUT]], Generic[ANY_MAPPING_INPUT]):
    """
    Objective that measures the hardware size (core count).
    Supports relative evaluation against a baseline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _graph_cache: Dict[int, Any] = PrivateAttr(default_factory=dict)

    def evaluate(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> float:
        """Returns the total number of cores in the hardware configuration."""
        input_id = id(state.mapping_input)
        if input_id not in self._graph_cache:
            self._graph_cache[input_id] = self._precompute_graph(state)

        data = self._graph_cache[input_id]
        e_valid = compute_e_valid(state, data=data)
        return compute_total_hw_connections(state.hw) - e_valid

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT], baseline: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> float:
        """Returns the ratio of state cores to baseline cores."""
        baseline_size = self.evaluate(baseline)
        if baseline_size <= 0:
            return float('inf')
        return self.evaluate(state) / baseline_size


    def _precompute_graph(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> Dict[str, Any]:
        graph = state.mapping_input.graph
        return {
            'N': graph.num_vertices(),
            'm': graph.num_edges(),
            'in_edges': [[int(src) for src in v.in_neighbors()] for v in graph.vertices()]
        }
