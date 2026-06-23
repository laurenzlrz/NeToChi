

class NeToChiException(Exception):
    """Base exception class for all NeToChi errors."""
    pass

class FrozenError(NeToChiException, AttributeError):
    """Raised when trying to modify a frozen object."""
    pass


class InputGenerationError(NeToChiException):
    """Base class for errors occurring during input generation."""
    pass

class DimensionError(InputGenerationError, ValueError):
    """Exception raised when an input array has incorrect dimensions."""
    pass

class NotSetError(InputGenerationError, ValueError):
    """Exception raised when a required value is not set."""
    pass

class InvalidAssignmentError(InputGenerationError, ValueError):
    """Exception raised when a pre-assignment is invalid."""
    pass

class InvalidConfigError(InputGenerationError, ValueError):
    """Exception raised when a configuration is invalid."""
    pass


class MappingError(NeToChiException):
    """Base class for errors occurring during the mapping process."""
    pass


class MappingValidationError(MappingError):
    """Raised when a mapping state or input fails validation checks."""
    pass


class HardwareConstraintError(MappingError):
    """Raised when a hardware constraint is violated during mapping."""
    pass


class PipelineError(NeToChiException):
    """Base exception for pipeline-related failures."""
    pass


class MapperError(PipelineError):
    """Raised when a mapper fails during execution."""
    pass


class MetricError(PipelineError):
    """Raised when a metric evaluation fails."""
    pass


class BaselineError(PipelineError):
    """Raised when a baseline provider fails."""
    pass


class ObjectiveError(NeToChiException):
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
