from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicMappingInput


class RandomMapper(BaseModel, BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper that assigns nodes to cores and addresses randomly.
    """
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        """Initialize state randomly for the given fixed hardware."""
        # Use unified state helper methods to avoid code duplication and magic logic
        state: MosaicNetworkMappingState = MosaicNetworkMappingState.from_input_random(mapping_input, self.seed)
        return state
