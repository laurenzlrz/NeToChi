from netochi.mapping.likelihood_state import MappingState

class LogLikelihoodMetric:
    def get_name(self) -> str:
        return self.__class__.__name__

    def evaluate(self, state: MappingState) -> float:
        return state.log_likelihood()

class InconsistenciesMetric:
    def get_name(self) -> str:
        return self.__class__.__name__

    def evaluate(self, state: MappingState) -> float:
        return state.inconsistencies()
        
class ConsistencyPercentageMetric:
    def get_name(self) -> str:
        return self.__class__.__name__

    def evaluate(self, state: MappingState) -> float:
        stats = state.mapping_stats()
        return stats.consistency_pct
