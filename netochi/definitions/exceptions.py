

class NeToChiException(Exception):
    """Base exception class for all NeToChi errors."""
    pass

class DimensionError(NeToChiException, ValueError):
    """Exception raised when an input array has incorrect dimensions."""
    pass

class NotSetError(NeToChiException, ValueError):
    """Exception raised when a required value is not set."""
    pass

class InvalidAssignmentError(NeToChiException, ValueError):
    """Exception raised when a pre-assignment is invalid."""
    pass

class InvalidConfigError(NeToChiException, ValueError):
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
