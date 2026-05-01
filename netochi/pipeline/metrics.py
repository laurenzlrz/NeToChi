from netochi.pipeline.core import BaseMetric
from netochi.mapping.likelihood_state import MappingState

class LogLikelihoodMetric(BaseMetric):
    def evaluate(self, state: MappingState) -> float:
        return state.log_likelihood()

class InconsistenciesMetric(BaseMetric):
    def evaluate(self, state: MappingState) -> float:
        return state.inconsistencies()
        
class ConsistencyPercentageMetric(BaseMetric):
    def evaluate(self, state: MappingState) -> float:
        stats = state.mapping_stats()
        return stats["consistency_pct"]
