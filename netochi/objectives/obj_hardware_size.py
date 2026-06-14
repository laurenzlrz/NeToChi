from netochi.mapping.interfaces import MosaicHWMappingState
from typing import Any
from netochi.objectives.interfaces import MappingObjective
from typing import Any

from netochi.mapping.interfaces import MosaicHWMappingState
from netochi.objectives.interfaces import MappingObjective


class MosaicHardwareSizeObjective(MappingObjective[MosaicHWMappingState[Any], MosaicHWMappingState[Any]]):
    """
    Objective that measures the hardware size (core count).
    Supports relative evaluation against a baseline.
    """
    def evaluate(self, state: MosaicHWMappingState[Any]) -> float:
        """Returns the total number of cores in the hardware configuration."""
        # TODO: Improve calculation to mirror more closely hardware costs
        hw = state.hw_to_evaluate
        penalty = hw.neurons_per_core * hw.total_cores

        return float(penalty)

    def evaluate_against_baseline(self, state: MosaicHWMappingState[Any], baseline: MosaicHWMappingState[Any]) -> float:
        """Returns the ratio of state cores to baseline cores."""
        baseline_size = self.evaluate(baseline)
        if baseline_size <= 0:
            return float('inf')
        return self.evaluate(state) / baseline_size
