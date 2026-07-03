from typing import Any, TYPE_CHECKING
from eventbus.event_bus import EventBus, TypeToken
from netochi.objectives.interfaces import ObjectiveResult, CustomObjectiveResult

if TYPE_CHECKING:
    from netochi.pipeline.pipeline_consumer.event_processor import ObjectiveResultWrapperEvent


class ObjectiveConsumer[RESULT: ObjectiveResult]:
    # Subclasses should override this with the specific ObjectiveResult subclass type
    RESULT_TYPE: type[ObjectiveResult] = ObjectiveResult

    def __init__(self, name: str):
        self.name = name

    def sign_in(self, event_bus: EventBus):
        """
        Subscribes to the event bus to listen for events of type RESULT_TYPE.
        """
        from netochi.pipeline.pipeline_consumer.event_processor import ObjectiveResultWrapperEvent
        subscription = TypeToken(ObjectiveResultWrapperEvent[Any, Any, Any, self.RESULT_TYPE])
        event_bus.subscribe_class(subscription, self.call)

    def call(self, event: "ObjectiveResultWrapperEvent[Any, Any, Any, RESULT]"):
        """
        Callback function that is called when an event is published.
        """
        pass

    def execute(self):
        """
        Executes after a run has been completed.
        """
        pass


class CustomObjectiveConsumer(ObjectiveConsumer[CustomObjectiveResult]):
    RESULT_TYPE = CustomObjectiveResult

    def __init__(self, name: str):
        super().__init__(name)
        self.received_results: list[CustomObjectiveResult] = []

    def call(self, event: "ObjectiveResultWrapperEvent[Any, Any, Any, CustomObjectiveResult]"):
        result = event.result
        print(f"[CustomObjectiveConsumer {self.name}] Received custom result: "
              f"name={result.objective_name}, value={result.value}, message={result.custom_message}", flush=True)
        self.received_results.append(result)

    def execute(self):
        print(f"[CustomObjectiveConsumer {self.name}] Total custom results consolidated: {len(self.received_results)}", flush=True)