from pydantic import BaseModel, Field
import icontract
from typing import Any

from netochi.definitions.constants import NAME_OBJ_HARDWARE_SIZE
from netochi.mapping.interfaces import MosaicHWMappingState
from netochi.objectives.interfaces import MappingObjective, AbstractObjectiveConfig


class MosaicHardwareSizeObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "MosaicHardwareSizeObjective":
        return MosaicHardwareSizeObjective(config=self)


class MosaicHardwareSizeObjective(MappingObjective[MosaicHWMappingState[Any], MosaicHWMappingState[Any]]):
    """
    Objective that measures the hardware size (core count).
    Supports relative evaluation against a baseline.
    """

    @icontract.require(lambda config: isinstance(config, MosaicHardwareSizeObjectiveConfig))
    def __init__(self, config: MosaicHardwareSizeObjectiveConfig) -> None:
        self.config = config
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

    def get_name(self) -> str:
        return NAME_OBJ_HARDWARE_SIZE


class MosaicHardwareSizeObjectiveConfig2(MosaicHardwareSizeObjectiveConfig):

    alpha: float = Field(gt=0, description="Weight for core area cost")
    beta: float = Field(gt=0, description="Weight for router area cost")
    gamma: float = Field(default=0, gt=0, description="Weight for utilization penalty")

    def create(self) -> "MosaicHardwareSizeObjective":
        return MosaicHardwareSizeObjective(config=self)


class MosaicHardwareSizeObjective2(MosaicHardwareSizeObjective):

    def __init__(self, config: MosaicHardwareSizeObjectiveConfig2) -> None:
        super().__init__(config=config)
        self.config = config

    def evaluate(self, state: MosaicHWMappingState[Any]) -> float:
        hw = state.hw_to_evaluate

        K = hw.total_cores
        Nc = hw.neurons_per_core
        Nr = hw.nodes_per_router
        graph = state.mapping_input.graph
        core_area_cost = self.config.alpha * (K * Nc * Nc)
        router_area_cost = self.config.beta * (Nc * ((Nr - 1) / (K - 1)))
        num_neurons = graph.num_vertices()

        wasted_space = (K * Nc) - num_neurons
        utilization_penalty = self.config.gamma * max(0, wasted_space)
        return core_area_cost + router_area_cost + utilization_penalty
