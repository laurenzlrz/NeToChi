from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from netochi.pipeline.interfaces import MappingStateConsumer
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import BaseMosaicMappingState
from netochi.pipeline.config import PipelineOutputConfig
from tests.utils_mapping_output_validation import validate_mosaic_mapping


class Validator(BaseModel, MappingStateConsumer[BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]):

    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    config: PipelineOutputConfig = Field(description="Configuration for validation behavior.")

    def consume(self, state: BaseMosaicMappingState[MosaicMappingInput], baseline: Optional[BaseMosaicMappingState[MosaicMappingInput]] = None) -> None:
        try:
            validate_mosaic_mapping(state.hw_to_evaluate, state)
            self.config.print_console("Validation successful!", name="validation_metric")
        except Exception as e:
            self.config.print_console(str(e), name="validation_metric")
