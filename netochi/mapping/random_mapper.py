from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicHWMappingInput


class RandomMapper(BaseModel, BaseMapper[MosaicNetworkMappingState[Any], MosaicHWMappingInput[Any]]):
    """
    Mapper that assigns nodes to cores and addresses randomly.
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)
    seed: Optional[int] = Field(default=None)

    def run(self, mapping_input: MosaicHWMappingInput[Any]) -> MosaicNetworkMappingState[Any]:
        """Initialize state randomly for the given fixed hardware."""
        # Use unified state helper methods to avoid code duplication and magic logic
        state: MosaicNetworkMappingState[Any] = MosaicNetworkMappingState.from_input(mapping_input)
        state.init_random_assignments(seed=self.seed)
        return state
