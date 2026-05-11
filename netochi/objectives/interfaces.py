from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict
from netochi.mapping.interfaces import MappingState

# -----------------------------------------------------------------------------
# Type Variables for Objectives
# -----------------------------------------------------------------------------
MAPPING_STATE = TypeVar('MAPPING_STATE', bound=MappingState)
MAPPING_STATE2 = TypeVar('MAPPING_STATE2', bound=MappingState)

# -----------------------------------------------------------------------------
# Objective Interfaces
# -----------------------------------------------------------------------------

class MappingObjective(BaseModel, ABC, Generic[MAPPING_STATE, MAPPING_STATE2]):
    """
    Abstract base class for all mapping objectives.
    Supports both direct evaluation and baseline-comparative evaluation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    @abstractmethod
    def evaluate(self, state: MAPPING_STATE) -> float:
        """
        Evaluate a mapping state independently.
        Returns a cost (lower is better).
        """
        pass

    @abstractmethod
    def evaluate_against_baseline(self, state: MAPPING_STATE, baseline: MAPPING_STATE2) -> float:
        """
        Evaluate a mapping state relative to a baseline reference state.
        """
        pass

class NetworkMappingObjective(MappingObjective[MAPPING_STATE, MAPPING_STATE2], ABC, Generic[MAPPING_STATE, MAPPING_STATE2]):
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
