from abc import ABC, abstractmethod
from typing import Generic
from pydantic import BaseModel, ConfigDict
from netochi.mapping.interfaces import MAPPING_STATE

class MappingObjective(BaseModel, ABC, Generic[MAPPING_STATE]):
    """Abstract base class for all mapping objectives."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    @abstractmethod
    def evaluate(self, state: MAPPING_STATE) -> float:
        """Evaluate a mapping state and return a cost (lower is better)."""
        pass

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the objective class."""
        return cls.__name__
