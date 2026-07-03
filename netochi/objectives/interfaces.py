from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Optional

from pydantic import BaseModel, ConfigDict

from netochi.mapping.interfaces import MappingState

# -----------------------------------------------------------------------------
# Objective Interfaces
# -----------------------------------------------------------------------------

class ObjectiveResult:

    def __init__(self, objective_name: str, value: float, baseline_value: Optional[float] = None):
        self.objective_name = objective_name
        self.value = value
        self.baseline_value = baseline_value


class CustomObjectiveResult(ObjectiveResult):

    def __init__(self, objective_name: str, value: float, baseline_value: Optional[float] = None, custom_message: str = "Default message"):
        super().__init__(objective_name, value, baseline_value)
        self.custom_message = custom_message


class AbstractObjectiveConfig(BaseModel, ABC):
    """Abstract base configuration class for mapping objectives."""
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    @abstractmethod
    def create(self) -> "MappingObjective[Any, Any]":
        """Instantiates the corresponding MappingObjective class."""
        pass


class MappingObjective[MAPPING_STATE: MappingState, MAPPING_STATE_BASELINE: MappingState](ABC):
    """
    Base class for all mapping objectives.
    Supports both direct evaluation and baseline-comparative evaluation.
    """
    
    @abstractmethod
    def evaluate(self, state: MAPPING_STATE, baseline: Optional[MAPPING_STATE_BASELINE] = None) -> ObjectiveResult:
        """
        Evaluate a mapping state, potentially against a baseline.
        Returns an ObjectiveResult containing absolute value and optional baseline_value.
        """
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class ObjectiveInterface[MAPPING_STATE: MappingState]:
    """
    Interface for objectives that compute log-likelihoods.
    This is a separate class to prevent information leakage
    """
    @abstractmethod
    def evaluate(self, state: MAPPING_STATE, baseline: Optional[MappingState] = None) -> ObjectiveResult:
        """Evaluate the log-likelihood of the mapping state."""
        pass
