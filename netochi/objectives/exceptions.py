"""
Custom exceptions for the neuromorphic mapping objectives.
"""

class ObjectiveError(Exception):
    """Base class for all objective-related errors."""
    pass

class BaselineMismatchError(ObjectiveError):
    """Raised when a baseline state is incompatible with the state being evaluated."""
    def __init__(self, message: str = "Baseline state does not match the network topology of the evaluated state."):
        super().__init__(message)

class EvaluationError(ObjectiveError):
    """Raised when an objective evaluation fails due to missing properties or invalid state."""
    pass

class HardwareConfigError(EvaluationError):
    """Raised when hardware configuration is missing or invalid for a specific objective."""
    pass
