from abc import ABC, abstractmethod
from typing import Optional, Any

from pydantic import BaseModel, Field, ConfigDict
import icontract

from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput
from netochi.mapping.interfaces import MappingState, BaseMosaicMappingState, MosaicNetworkMappingState, BaseMapper
from netochi.mapping.random_mapper import RandomMapperConfig
from .evaluator_bundle import BaselineStorer


class AdapterBaselineProvider[MAPPING_INPUT: MappingInput, BASELINE_STATE: MappingState](BaselineStorer[MAPPING_INPUT, BASELINE_STATE]):
    """Adapter to use a provided mapping state as a baseline."""

    mapper: BaseMapper[BASELINE_STATE, MAPPING_INPUT] = Field(description="Mapper to generate the baseline state.")

    def __init__(self, mapper: BaseMapper[BASELINE_STATE, MAPPING_INPUT]):
        super().__init__()
        self.mapper = mapper

    def precompute_baseline(self, mapping_input: MAPPING_INPUT) -> None:
        self._baseline_state = self.mapper.run(mapping_input)


class RandomMosaicBaselineProviderConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    seed: Optional[int] = Field(default=None, description="Random seed.")

    def create(self) -> "RandomMosaicBaselineProvider":
        return RandomMosaicBaselineProvider(config=self)


class RandomMosaicBaselineProvider(AdapterBaselineProvider[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput]]):
    """Baseline provider that uses a random mapper to generate the baseline state."""

    @icontract.require(lambda config: isinstance(config, RandomMosaicBaselineProviderConfig))
    def __init__(self, config: RandomMosaicBaselineProviderConfig):
        self.config = config
        super().__init__(mapper=RandomMapperConfig(seed=config.seed).create())


class MosaicGroundTruthBaselineProviderConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    mapper: Optional[BaseMapper[BaseMosaicMappingState[MosaicMappingInput], MosaicMappingInput]] = Field(default=None, description="Fallback mapper if GT is not available.")

    def create(self) -> "MosaicGroundTruthBaselineProvider":
        return MosaicGroundTruthBaselineProvider(config=self)


class MosaicGroundTruthBaselineProvider(AdapterBaselineProvider[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput]]):
    """Extracts ground truth from MosaicMappingInput if available."""

    @icontract.require(lambda config: isinstance(config, MosaicGroundTruthBaselineProviderConfig))
    def __init__(self, config: MosaicGroundTruthBaselineProviderConfig):
        self.config = config
        mapper = config.mapper
        if mapper is None:
            mapper = RandomMapperConfig(seed=42).create()  # Default to a random mapper if no specific mapper is provided
        super().__init__(mapper=mapper)

    def precompute_baseline(self, mapping_input: MosaicMappingInput) -> None:
        state = MosaicNetworkMappingState.from_input_zero(mapping_input)
        if mapping_input.assignment is None:
            return super().precompute_baseline(mapping_input)
        assert mapping_input.assignment is not None, "MosaicMappingInput must contain an assignment for ground truth baseline."
        state.unfreeze()
        state.assignment = mapping_input.assignment
        state.freeze()
        self._baseline_state = state