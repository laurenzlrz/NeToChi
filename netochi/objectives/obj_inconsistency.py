from typing import Dict, Any

from pydantic import BaseModel, PrivateAttr, ConfigDict

from netochi.definitions.exceptions import BaselineMismatchError
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
)
from netochi.objectives.interfaces import MappingObjective
from netochi.objectives.utils import compute_e_valid


class InconsistencyObjectiveFabric(BaseModel, MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    _graph_cache: Dict[int, Any] = PrivateAttr(default_factory=dict)

    def _preload_graph(self, state: BaseMosaicMappingState) -> Dict[str, Any]:
        """Preload graph data into cache for faster evaluation."""
        input_id = id(state.mapping_input)
        if input_id not in self._graph_cache:
            graph = state.mapping_input.graph
            self._graph_cache[input_id] = {
                'N': graph.num_vertices(),
                'm': graph.num_edges(),
                'in_edges': [[int(src) for src in v.in_neighbors()] for v in graph.vertices()]
            }
        return self._graph_cache[input_id]


class InconsistencyObjective(InconsistencyObjectiveFabric):
    """
    Objective that counts the number of invalid edges (edges violating hardware constraints).
    """


    def evaluate(self, state: BaseMosaicMappingState[Any]) -> float:
        """Returns total number of invalid edges."""
        return float(compute_e_valid(state, self._preload_graph(state)))

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[Any], baseline: BaseMosaicMappingState[Any]) -> float:
        return self.evaluate(state) - self.evaluate(baseline)

    def get_name(self) -> str:
        return "Inconsistencies"


class InconsistencyRelativeObjective(InconsistencyObjectiveFabric):
    """
    Objective that counts the number of invalid edges (edges violating hardware constraints) relative to the total number of edges
    """

    def evaluate(self, state: BaseMosaicMappingState[Any]) -> float:
        """Returns ratio of invalid edges to total edges."""
        inconsistencies = float(compute_e_valid(state, self._preload_graph(state)))
        return float(inconsistencies) / float(self._graph_cache[id(state.mapping_input)]['m'])

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[Any], baseline: BaseMosaicMappingState[Any]) -> float:
        return self.evaluate(state) / self.evaluate(baseline)

    def get_name(self) -> str:
        return "Inconsistencies (Fraction)"


class InconsistencyPercentageMetric(InconsistencyObjectiveFabric):
    """
    Evaluates an inconsistency-based objective and returns it as a percentage of total edges.
    """

    def evaluate(self, state: BaseMosaicMappingState[Any]) -> float:
        invalid_edges = float(compute_e_valid(state, self._preload_graph(state)))
        total_edges = state.mapping_input.graph.num_edges()

        if total_edges == 0:
            return 0.0
        return (float(invalid_edges) / float(total_edges)) * 100.0

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[Any], baseline: BaseMosaicMappingState[Any]) -> float:
        baseline_val = self.evaluate(baseline)
        if baseline_val == 0:
            return 1.0 if self.evaluate(state) == 0 else 100.0
        return self.evaluate(state) / baseline_val

    def get_name(self) -> str:
        return "Inconsistencies (%)"
