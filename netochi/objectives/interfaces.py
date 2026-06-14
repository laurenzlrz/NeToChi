from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any
from pydantic import BaseModel, ConfigDict

from netochi.mapping.interfaces import MappingState

# -----------------------------------------------------------------------------
# Objective Interfaces
# -----------------------------------------------------------------------------

class MappingObjective[MAPPING_STATE: MappingState, MAPPING_STATE_BASELINE: MappingState](ABC):
    """
    Base class for all mapping objectives.
    Supports both direct evaluation and baseline-comparative evaluation.
    """
    
    def evaluate(self, state: MAPPING_STATE) -> float:
        """
        Evaluate a mapping state independently.
        Returns a cost (lower is better).
        """
        raise NotImplementedError

    def evaluate_against_baseline(self, state: MAPPING_STATE, baseline: MAPPING_STATE_BASELINE) -> float:
        """
        Evaluate a mapping state relative to a baseline reference state.
        """
        raise NotImplementedError


class ObjectiveInterface[MAPPING_STATE: MappingState]:
    """
    Interface for objectives that compute log-likelihoods.
    This is a separate class to prevent information leakage
    """
    def evaluate(self, state: MAPPING_STATE) -> float:
        """Evaluate the log-likelihood of the mapping state."""
        raise NotImplementedError
