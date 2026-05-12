"""
Custom exceptions for the neuromorphic mapping pipeline.
"""

class PipelineError(Exception):
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