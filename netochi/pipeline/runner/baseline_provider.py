from abc import ABC, abstractmethod
from typing import Optional, Any

from pydantic import Field

from netochi.input_generator.interfaces import MappingInput, MosaicMappingInput
from netochi.mapping.interfaces import MappingState, BaseMosaicMappingState, MosaicNetworkMappingState, BaseMapper
from netochi.mapping.random_mapper import RandomMapper
from .evaluator_bundle import BaselineStorer



class AdapterBaselineProvider[MAPPING_INPUT: MappingInput, BASELINE_STATE: MappingState](BaselineStorer[MAPPING_INPUT, BASELINE_STATE]):
    """Adapter to use a provided mapping state as a baseline."""

    mapper: BaseMapper[BASELINE_STATE, MAPPING_INPUT] = Field(description="Mapper to generate the baseline state.")

    def __init__(self, mapper: BaseMapper[BASELINE_STATE, MAPPING_INPUT]):
        super().__init__()
        self.mapper = mapper

    def precompute_baseline(self, mapping_input: MAPPING_INPUT) -> None:
        self._baseline_state = self.mapper.run(mapping_input)


class RandomMosaicBaselineProvider(AdapterBaselineProvider[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput]]):
    """Baseline provider that uses a random mapper to generate the baseline state."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(mapper=RandomMapper(seed=seed))


class MosaicGroundTruthBaselineProvider(AdapterBaselineProvider[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput]]):
    """Extracts ground truth from MosaicMappingInput if available."""

    def __init__(self, mapper: Optional[BaseMapper[BaseMosaicMappingState[MosaicMappingInput], MosaicMappingInput]] = None):
        if mapper is None:
            mapper = RandomMapper(seed=42)  # Default to a random mapper if no specific mapper is provided
        super().__init__(mapper=mapper)

    def precompute_baseline(self, mapping_input: MosaicMappingInput) -> None:
        state = MosaicNetworkMappingState.from_input_zero(mapping_input)
        if mapping_input.assignment is None:
            return super().precompute_baseline(mapping_input)
        assert mapping_input.assignment is not None, "MosaicMappingInput must contain an assignment for ground truth baseline."
        state.assignment = mapping_input.assignment
        self._baseline_state = state