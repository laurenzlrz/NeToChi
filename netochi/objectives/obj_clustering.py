from typing import Any

import icontract
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from netochi.definitions.constants import NAME_OBJ_CLUSTERING_ARI, NAME_OBJ_CLUSTERING_ACCURACY
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
)
from netochi.objectives.interfaces import MappingObjective, AbstractObjectiveConfig


class ClusteringARIObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "":
        return ClusteringARIObjective(config=self)


class ClusteringAccuracyObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "":
        return ClusteringAccuracyObjective(config=self)



class ClusteringARIObjective(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]):
    """
    Objective that computes the adjusted rand score (= proportion of pairs of nodes that were correctly kept together or correctly separated)
    """

    @icontract.require(lambda config: isinstance(config, ClusteringARIObjectiveConfig))
    def __init__(self, config: ClusteringARIObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[MosaicMappingInput]) -> float:
        if state.mapping_input.assignment is None:
            return float('nan')
        true_labels = state.mapping_input.assignment.neuron_core_pre_assignment
        predicted_labels = state.c
        ari = adjusted_rand_score(true_labels, predicted_labels)
        return ari

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[MosaicMappingInput], baseline: BaseMosaicMappingState[Any]) -> float:
        return self.evaluate(state=state)

    def get_name(self) -> str:
        return NAME_OBJ_CLUSTERING_ARI


class ClusteringAccuracyObjective(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]):
    """
    Objective that measures how many neurons were correctly clustered.
    1. uses Hungarian algorithm to find the optimal 1-to-1 mapping from predicted to ground truth clusters
    2. computes fraction of nodes that are in correct cluster
    """

    @icontract.require(lambda config: isinstance(config, ClusteringAccuracyObjectiveConfig))
    def __init__(self, config: ClusteringAccuracyObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[Any]) -> float:
        if state.mapping_input.assignment is None:
            return float('nan')
        true_labels = state.mapping_input.assignment.neuron_core_pre_assignment
        predicted_labels = state.c
        cm = confusion_matrix(true_labels, predicted_labels)
        # Use the Hungarian algorithm to find optimal 1-to-1 mapping
        row_ind, col_ind = linear_sum_assignment(-cm) # We pass -cm because the algorithm minimizes cost, but we want to maximize overlap
        accuracy = cm[row_ind, col_ind].sum() / np.size(true_labels)
        return accuracy

    def evaluate_against_baseline(self, state: BaseMosaicMappingState[MosaicMappingInput], baseline: BaseMosaicMappingState[Any]) -> float:
        return self.evaluate(state=state)

    def get_name(self) -> str:
        return NAME_OBJ_CLUSTERING_ACCURACY
