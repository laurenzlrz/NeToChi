from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from netochi.pipeline import PipelineSummary


class PipelineConsumer(ABC, BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)

    @abstractmethod
    def consume(self, data: PipelineSummary) -> None:
        """
        Consume the pipeline summary data.
        This method should be implemented by subclasses to define specific consumption behavior.
        """
        pass