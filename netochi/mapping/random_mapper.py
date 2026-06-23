from typing import Optional
from pydantic import BaseModel, Field
import icontract

from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import MosaicMappingInput


class RandomMapperConfig(BaseModel):
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")

    def create(self) -> "RandomMapper":
        return RandomMapper(config=self)


class RandomMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper that assigns nodes to cores and addresses randomly.
    """
    @icontract.require(lambda config: isinstance(config, RandomMapperConfig))
    def __init__(self, config: RandomMapperConfig) -> None:
        self.config = config
        self.seed = config.seed

    def run(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        """Initialize state randomly for the given fixed hardware."""
        # Use unified state helper methods to avoid code duplication and magic logic
        state: MosaicNetworkMappingState = MosaicNetworkMappingState.from_input_random(mapping_input, self.seed)
        return state
