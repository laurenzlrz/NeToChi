from abc import ABC, abstractmethod
from netochi.mapping.likelihood_state import MappingState

class MappingObjective(ABC):
    """Abstract base class for all mapping objectives."""
    @abstractmethod
    def evaluate(self, state: MappingState) -> float:
        """Evaluate a mapping state and return a cost (lower is better)."""
        pass

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the objective class."""
        return cls.__name__

class LikelihoodObjective(MappingObjective):
    """Objective minimizing the negative log-likelihood for fixed hardware."""
    def evaluate(self, state: MappingState) -> float:
        """Return negative log-likelihood of the state."""
        return -state.log_likelihood()

class HardwareCostObjective(MappingObjective):
    """Objective minimizing hardware size with a penalty for inconsistencies."""
    def __init__(self, hw_weight: float = 1.0, error_weight: float = 1000.0):
        """Initialize with weights for hardware size and error penalty."""
        self.hw_weight = hw_weight
        self.error_weight = error_weight

    def evaluate(self, state: MappingState) -> float:
        """Calculate total cost as weighted sum of cores and negative LL."""
        inconsistencies_cost = -state.log_likelihood()
        hw_size = state.config.total_cores
        return (self.hw_weight * hw_size) + (self.error_weight * inconsistencies_cost)

    @classmethod
    def get_name(cls) -> str:
        """Return human-readable name of the objective."""
        return "HardwareCostObjective"
