class DimensionError(ValueError):
    """Exception raised when an input array has incorrect dimensions."""
    pass

class NotSetError(ValueError):
    """Exception raised when a required value is not set."""
    pass

class InvalidAssignmentError(ValueError):
    """Exception raised when a pre-assignment is invalid."""
    pass

class InvalidConfigError(ValueError):
    """Exception raised when a configuration is invalid."""
    pass

