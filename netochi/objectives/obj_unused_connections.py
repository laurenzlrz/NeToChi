from typing import Any

from netochi.mapping.interfaces import BaseMosaicMappingState
from netochi.objectives.obj_inconsistency import InconsistencyObjectiveFabric
from netochi.objectives.utils import compute_e_valid, compute_total_hw_connections


class UnusedConnectionsObjective(InconsistencyObjectiveFabric):
    """
    Objective that measures the hardware size (core count).
    Supports relative evaluation against a baseline.
    """

    def evaluate(self, state: BaseMosaicMappingState[Any]) -> float:
        """Returns the total number of cores in the hardware configuration."""
        data = self._preload_graph(state)
        e_valid = compute_e_valid(state, data=data)
        return compute_total_hw_connections(state.hw_to_evaluate) - e_valid

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[Any], baseline: BaseMosaicMappingState[Any]) -> float:
        """Returns the ratio of state cores to baseline cores."""
        baseline_size = self.evaluate(baseline)
        if baseline_size <= 0:
            return float('inf')
        return self.evaluate(state) / baseline_size