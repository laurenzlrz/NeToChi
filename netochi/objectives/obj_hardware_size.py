from typing import Generic, Any
from netochi.objectives.interfaces import MappingObjective
from netochi.mapping.interfaces import BaseMosaicMappingState, ANY_MAPPING_INPUT

class MosaicHardwareSizeObjective(MappingObjective[BaseMosaicMappingState[ANY_MAPPING_INPUT], BaseMosaicMappingState[ANY_MAPPING_INPUT]], Generic[ANY_MAPPING_INPUT]):
    """
    Objective that measures the hardware size (core count).
    Supports relative evaluation against a baseline.
    """
    def evaluate(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> float:
        """Returns the total number of cores in the hardware configuration."""
        # Using core count scaled by network size as requested by TODO
        num_neurons = state.mapping_input.graph.num_vertices()
        return float(state.hw.total_cores * num_neurons)

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[ANY_MAPPING_INPUT], baseline: BaseMosaicMappingState[ANY_MAPPING_INPUT]) -> float:
        """Returns the ratio of state cores to baseline cores."""
        baseline_size = self.evaluate(baseline)
        if baseline_size <= 0:
            return float('inf')
        return self.evaluate(state) / baseline_size

    def get_name(self) -> str:
        return "Hardware Size"

