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

class NetworkMappingObjective(MappingObjective[MAPPING_STATE, MAPPING_STATE2], Generic[MAPPING_STATE, MAPPING_STATE2]):
    """
    Objectives that evaluate network assignments relative to a baseline mapping state.
    """
    pass

# -----------------------------------------------------------------------------
# Capability Interfaces
# -----------------------------------------------------------------------------

class LogLikelihoodObjectiveInterface(BaseModel, Generic[MAPPING_STATE]):
    """Specific contract for log-likelihood based evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    def log_likelihood(self, state: MAPPING_STATE) -> float:
        """Returns log-likelihood of state."""
        raise NotImplementedError
