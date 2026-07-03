from typing import Dict, Any, Optional
import icontract

from netochi.definitions.constants import NAME_OBJ_VALID_EDGES
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
)
from netochi.objectives.interfaces import MappingObjective, AbstractObjectiveConfig, ObjectiveResult
from netochi.objectives.utils import compute_e_valid


class ValidEdgesObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "ValidEdgesObjective":
        return ValidEdgesObjective(config=self)


class ValidEdgesObjectiveFabric(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]):
    def __init__(self) -> None:
        self._graph_cache: Dict[int, Any] = {}

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


class ValidEdgesObjective(ValidEdgesObjectiveFabric):
    """
    Objective that counts the number of valid edges (edges fulfilling hardware constraints).
    """

    @icontract.require(lambda config: isinstance(config, ValidEdgesObjectiveConfig))
    def __init__(self, config: ValidEdgesObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[Any], baseline: Optional[BaseMosaicMappingState[Any]] = None) -> ObjectiveResult:
        """Returns total number of valid edges."""
        val = float(compute_e_valid(state, self._preload_graph(state)))
        if baseline is not None:
            baseline_val = float(compute_e_valid(baseline, self._preload_graph(baseline)))
            diff = val - baseline_val
        else:
            diff = None
        return ObjectiveResult(self.get_name(), val, diff)

    def get_name(self) -> str:
        return NAME_OBJ_VALID_EDGES
