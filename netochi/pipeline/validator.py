from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
import icontract

from netochi.pipeline.interfaces import MappingStateConsumer
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import BaseMosaicMappingState
from netochi.pipeline.config import PipelineOutput
from tests.utils_mapping_output_validation import validate_mosaic_mapping


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    pipeline_output: PipelineOutput = Field(..., description="Pipeline output manager.")

    def create(self) -> "Validator":
        return Validator(config=self)


class Validator(MappingStateConsumer[BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]):

    @icontract.require(lambda config: isinstance(config, ValidatorConfig))
    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config
        self.pipeline_output = config.pipeline_output

    def consume(self, state: BaseMosaicMappingState[MosaicMappingInput], baseline: Optional[BaseMosaicMappingState[MosaicMappingInput]] = None) -> None:
        try:
            validate_mosaic_mapping(state.hw_to_evaluate, state)
            self.pipeline_output.print_console("Validation successful!", name="validation_metric")
        except Exception as e:
            self.pipeline_output.print_console(str(e), name="validation_metric")
