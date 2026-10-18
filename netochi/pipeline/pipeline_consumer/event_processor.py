from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass
from typing import Generic, Any, List
from netochi.pipeline.pipeline_consumer.objective_consumer import ObjectiveConsumer
from eventbus.event_bus import EventBus, BaseEvent, TypeToken
from netochi.definitions import freezable
from netochi.definitions.generics import BaselineState_contra, Input_contra, MappingState_contra, ObjectiveResult_co
from netochi.objectives.interfaces import ObjectiveResult
from netochi.pipeline import PipelineSummary, ExperimentResult
from netochi.pipeline.interfaces import PipelineConsumer

@freezable
class ObjectiveResultWrapperEvent(BaseEvent,
    Generic[Input_contra, MappingState_contra, BaselineState_contra, ObjectiveResult_co]):
    """
    Event wrapper for ObjectiveResult.
    This class can be extended to include additional event-related information as needed.
    """
    def __init__(self, result: ObjectiveResult_co,
                 exp_result: ExperimentResult[Input_contra, MappingState_contra, BaselineState_contra, Any],
                 summary: PipelineSummary[Input_contra, MappingState_contra, BaselineState_contra]) -> None:
        super().__init__(type_token=TypeToken(ObjectiveResultWrapperEvent[Any, Any, Any, type(result)]))
        self.result = result
        self.exp_result = exp_result
        self.summary = summary

class EventProcessorConfig(BaseModel):
    """
    Configuration for the EventProcessor.
    This class can be extended to include additional configuration parameters as needed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    objective_consumer: List[ObjectiveConsumer[Any]] = Field(default_factory=list)

class EventProcessor(PipelineConsumer[Input_contra, MappingState_contra, BaselineState_contra], Generic[Input_contra, MappingState_contra, BaselineState_contra]):

    def __init__(self, config: EventProcessorConfig) -> None:
        """
        Initialize the EventProcessor.
        This class is designed to process events from the pipeline summary.
        """
        super().__init__()
        self._config = config
        self._event_bus = EventBus()
        self._consumers = config.objective_consumer
        for consumer in self._consumers:
            consumer.sign_in(self._event_bus)

    def consume(self, data: PipelineSummary[Input_contra, MappingState_contra, BaselineState_contra]) -> None:
        """
        Consume the pipeline summary data.
        This method must be implemented by subclasses.
        """
        for exp_result in data.results:
            for labeled_result in exp_result.objective_results:
                event = ObjectiveResultWrapperEvent(
                    result=labeled_result.result,
                    exp_result=exp_result,
                    summary=data
                )

                self._event_bus.publish(event)
                
        for consumer in self._consumers:
            consumer.execute()

