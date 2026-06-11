from typing import Generic

from netochi.input_generator.interfaces import WITH_HW_INPUT
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
    ANY_MAPPING_INPUT
)
from netochi.objectives import InconsistencyObjective


class InconsistencyRelativeObjective(InconsistencyObjective[BaseMosaicMappingState[ANY_MAPPING_INPUT], BaseMosaicMappingState[ANY_MAPPING_INPUT]], Generic[ANY_MAPPING_INPUT, WITH_HW_INPUT]):
    """
    Objective that counts the number of invalid edges (edges violating hardware constraints) relative to the total number of edges
    """

    def evaluate(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> float:
        """Returns total number of invalid edges."""
        inconsistencies = super().evaluate(state)
        return float(inconsistencies) / float(self._graph_cache[id(state.mapping_input)]['m'])

