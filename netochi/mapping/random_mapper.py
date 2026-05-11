from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.mcmc.likelihood_state import MappingState
from netochi.mapping.interfaces import BaseMapper, MosaicMappingState
from netochi.input_generator.interfaces import MosaicMappingInput


class RandomMapper(BaseModel, BaseMapper[MosaicMappingState, MosaicMappingInput[Any]]):
    """
    Mapper that assigns nodes to cores and addresses randomly.
    
    Refactored to follow the "Großprojekt" Pydantic standard.
    """
    model_config = ConfigDict(frozen=True)
    seed: Optional[int] = Field(default=None)

    def run(self, mapping_input: MosaicMappingInput[Any]) -> MosaicMappingState:
        """Initialize state randomly for the given fixed hardware."""
        graph = mapping_input.graph
        hw = mapping_input.hw_config
        
        calc_state = MappingState(graph=graph, config=hw)
        calc_state.init_random(seed=self.seed)
        
        return MosaicMappingState(
            mapping_input=mapping_input,
            neuron_core_idxs_assignment=calc_state.c,
            neuron_local_idxs_assignment=calc_state.x,
            neuron_slice_assignments=calc_state.s,
        )
