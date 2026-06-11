from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any
from pydantic import BaseModel, ConfigDict

from netochi.mapping.interfaces import BaseMosaicMappingState

# -----------------------------------------------------------------------------
# Type Variables for Objectives
# -----------------------------------------------------------------------------
MAPPING_STATE = TypeVar('MAPPING_STATE', bound=BaseMosaicMappingState[Any])
MAPPING_STATE2 = TypeVar('MAPPING_STATE2', bound=BaseMosaicMappingState[Any])

# -----------------------------------------------------------------------------
# Objective Interfaces
# -----------------------------------------------------------------------------

class MappingObjective(BaseModel, Generic[MAPPING_STATE, MAPPING_STATE2]):
    """
    Base class for all mapping objectives.
    Supports both direct evaluation and baseline-comparative evaluation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    def evaluate(self, state: MAPPING_STATE) -> float:
        """
        Evaluate a mapping state independently.
        Returns a cost (lower is better).
        """
        raise NotImplementedError

    def evaluate_against_baseline(self, state: MAPPING_STATE, baseline: MAPPING_STATE2) -> float:
        """
        Evaluate a mapping state relative to a baseline reference state.
        """
        raise NotImplementedError

    def get_name(self) -> str:
        return self.__class__.__name__


class NetworkMappingObjective(MappingObjective[MAPPING_STATE, MAPPING_STATE2], Generic[MAPPING_STATE, MAPPING_STATE2]):
    """
    Objectives that evaluate network assignments relative to a baseline mapping state.
    """
    pass

# -----------------------------------------------------------------------------
# Capability Interfaces
# -----------------------------------------------------------------------------

class LogLikelihoodObjectiveInterface(ABC, Generic[MAPPING_STATE]):
    """Specific contract for log-likelihood based evaluation."""
    
    @abstractmethod
    def log_likelihood(self, state: MAPPING_STATE) -> float:
        """Returns log-likelihood of state."""
        pass
