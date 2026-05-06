import graph_tool.all as gt
from netochi.mapping.hardware_config import HardwareConfig
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.core import FixedHardwareInput

class RandomMapper:
    """Mapper that assigns nodes to cores and addresses randomly."""
    
    def get_name(self) -> str:
        return self.__class__.__name__

    def map_fixed_hardware(self, mapping_input: FixedHardwareInput, seed=None) -> MappingState:
        """Initialize state randomly for the given fixed hardware."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        state.init_random(seed=seed)
        return state

