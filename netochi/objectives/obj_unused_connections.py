from typing import Any, Optional
import icontract

from netochi.definitions.constants import NAME_OBJ_UNUSED_CONNECTIONS
from netochi.mapping.interfaces import BaseMosaicMappingState
from netochi.objectives.obj_inconsistency import InconsistencyObjectiveFabric
from netochi.objectives.utils import compute_e_valid, compute_total_hw_connections
from netochi.objectives.interfaces import AbstractObjectiveConfig, ObjectiveResult


class UnusedConnectionsObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "UnusedConnectionsObjective":
        return UnusedConnectionsObjective(config=self)


class UnusedConnectionsObjective(InconsistencyObjectiveFabric):
    """
    Objective that measures the hardware size (core count).
    Supports relative evaluation against a baseline.
    """

    @icontract.require(lambda config: isinstance(config, UnusedConnectionsObjectiveConfig))
    def __init__(self, config: UnusedConnectionsObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[Any], baseline: Optional[BaseMosaicMappingState[Any]] = None) -> ObjectiveResult:
        """Returns the total number of cores in the hardware configuration."""
        data = self._preload_graph(state)
        e_valid = compute_e_valid(state, data=data)
        val = float(compute_total_hw_connections(state.hw_to_evaluate) - e_valid)

        if baseline is not None:
            baseline_val = self.evaluate(baseline).value
            if baseline_val <= 0:
                diff = float('inf')
            else:
                diff = val / baseline_val
        else:
            diff = None

        return ObjectiveResult(self.get_name(), val, diff)

    def get_name(self) -> str:
        return NAME_OBJ_UNUSED_CONNECTIONS
