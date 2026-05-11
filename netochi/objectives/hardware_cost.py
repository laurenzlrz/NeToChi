from typing import Generic
from pydantic import Field, ConfigDict
from netochi.objectives.interfaces import MappingObjective
from netochi.mapping.interfaces import MosaicHWMappingState, ANY_MAPPING_INPUT, PAYLOAD
from netochi.objectives.constants import DEFAULT_HW_WEIGHT


class HardwareCostObjective(MappingObjective[MosaicHWMappingState[ANY_MAPPING_INPUT, PAYLOAD], MosaicHWMappingState[ANY_MAPPING_INPUT, PAYLOAD]], Generic[ANY_MAPPING_INPUT, PAYLOAD]):
    """
    Objective that measures the hardware cost (resource usage) specifically for Mosaic hardware.
    """
    model_config = ConfigDict(frozen=True)

    hw_weight: float = Field(default=DEFAULT_HW_WEIGHT, description="Penalty weight per active core.")

    def evaluate(self, state: MosaicHWMappingState[ANY_MAPPING_INPUT, PAYLOAD]) -> float:
        """Returns the hardware cost based on total cores."""
        return float(state.hw.total_cores * self.hw_weight)

    def evaluate_against_baseline(self, state: MosaicHWMappingState[ANY_MAPPING_INPUT, PAYLOAD], baseline: MosaicHWMappingState[ANY_MAPPING_INPUT, PAYLOAD]) -> float:
        """Returns relative hardware cost compared to baseline."""
        baseline_cost = self.evaluate(baseline)
        if baseline_cost <= 0:
            return float('inf')
        return self.evaluate(state) / baseline_cost
