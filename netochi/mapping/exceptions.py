class NeToChiException(Exception):
    """Base exception class for all NeToChi errors."""
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
