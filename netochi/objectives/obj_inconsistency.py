from typing import Dict, Any, Optional
import icontract

from netochi.definitions.constants import NAME_OBJ_INCONSISTENCIES, NAME_OBJ_INCONSISTENCIES_FRACTION, \
    NAME_OBJ_INCONSISTENCIES_PERCENTAGE
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
)
from netochi.objectives.interfaces import MappingObjective, AbstractObjectiveConfig, ObjectiveResult
from netochi.objectives.utils import compute_inconsistencies, compute_e_valid


class InconsistencyObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "InconsistencyObjective":
        return InconsistencyObjective(config=self)


class InconsistencyRelativeObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "InconsistencyRelativeObjective":
        return InconsistencyRelativeObjective(config=self)


class InconsistencyPercentageMetricConfig(AbstractObjectiveConfig):
    def create(self) -> "InconsistencyPercentageMetric":
        return InconsistencyPercentageMetric(config=self)


class InconsistencyObjectiveFabric(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]):
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


class InconsistencyObjective(InconsistencyObjectiveFabric):
    """
    Objective that counts the number of invalid edges (edges violating hardware constraints).
    """

    @icontract.require(lambda config: isinstance(config, InconsistencyObjectiveConfig))
    def __init__(self, config: InconsistencyObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[Any], baseline: Optional[BaseMosaicMappingState[Any]] = None) -> ObjectiveResult:
        """Returns total number of invalid edges."""
        val = float(compute_inconsistencies(state, self._preload_graph(state)))
        if baseline is not None and baseline.mapping_input is not None:
            baseline_val = float(compute_inconsistencies(baseline, self._preload_graph(baseline)))
            diff = val - baseline_val
        else:
            diff = None
        return ObjectiveResult(self.get_name(), val, diff)

    def get_name(self) -> str:
        return NAME_OBJ_INCONSISTENCIES


class InconsistencyRelativeObjective(InconsistencyObjectiveFabric):
    """
    Objective that counts the number of invalid edges (edges violating hardware constraints) relative to the total number of edges
    """

    @icontract.require(lambda config: isinstance(config, InconsistencyRelativeObjectiveConfig))
    def __init__(self, config: InconsistencyRelativeObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[Any], baseline: Optional[BaseMosaicMappingState[Any]] = None) -> ObjectiveResult:
        """Returns ratio of invalid edges to total edges."""
        inconsistencies = float(compute_inconsistencies(state, self._preload_graph(state)))
        val = float(inconsistencies) / float(self._graph_cache[id(state.mapping_input)]['m'])
        
        if baseline is not None and baseline.mapping_input is not None:
            b_inconsistencies = float(compute_inconsistencies(baseline, self._preload_graph(baseline)))
            b = float(b_inconsistencies) / float(self._graph_cache[id(baseline.mapping_input)]['m'])
            if b == 0:
                diff = -1.0
            else:
                diff = val / b
        else:
            diff = None
            
        return ObjectiveResult(self.get_name(), val, diff)

    def get_name(self) -> str:
        return NAME_OBJ_INCONSISTENCIES_FRACTION


class InconsistencyPercentageMetric(InconsistencyObjectiveFabric):
    """
    Evaluates an inconsistency-based objective and returns it as a percentage of total edges.
    """

    @icontract.require(lambda config: isinstance(config, InconsistencyPercentageMetricConfig))
    def __init__(self, config: InconsistencyPercentageMetricConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[Any], baseline: Optional[BaseMosaicMappingState[Any]] = None) -> ObjectiveResult:
        invalid_edges = float(compute_inconsistencies(state, self._preload_graph(state)))
        total_edges = state.mapping_input.graph.num_edges()
        val = 0.0 if total_edges == 0 else (float(invalid_edges) / float(total_edges)) * 100.0
        
        if baseline is not None and baseline.mapping_input is not None:
            baseline_val = self.evaluate(baseline).value
            if baseline_val == 0:
                diff = 1.0 if val == 0 else 100.0
            else:
                diff = val / baseline_val
        else:
            diff = None
            
        return ObjectiveResult(self.get_name(), val, diff)

    def get_name(self) -> str:
        return NAME_OBJ_INCONSISTENCIES_PERCENTAGE
