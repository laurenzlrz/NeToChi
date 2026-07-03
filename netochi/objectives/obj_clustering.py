from typing import Any, Optional

import icontract
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from netochi.definitions.constants import NAME_OBJ_CLUSTERING_ARI, NAME_OBJ_CLUSTERING_ACCURACY, \
    NAME_OBJ_CORE_DISTANCE_DIFF
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
)
from netochi.objectives.interfaces import MappingObjective, AbstractObjectiveConfig, ObjectiveResult


class ClusteringARIObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "ClusteringARIObjective":
        return ClusteringARIObjective(config=self)


class ClusteringAccuracyObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "ClusteringAccuracyObjective":
        return ClusteringAccuracyObjective(config=self)


class CoreDistanceDifferenceObjectiveConfig(AbstractObjectiveConfig):
    def create(self) -> "CoreDistanceDifferenceObjective":
        return CoreDistanceDifferenceObjective(config=self)


class ClusteringARIObjective(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]):
    """
    Objective that computes the adjusted rand score (= proportion of pairs of pairs of nodes that were correctly kept together or correctly separated)
    """

    @icontract.require(lambda config: isinstance(config, ClusteringARIObjectiveConfig))
    def __init__(self, config: ClusteringARIObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[MosaicMappingInput], baseline: Optional[Any] = None) -> ObjectiveResult:
        if state is None or state.mapping_input is None or state.mapping_input.assignment is None:
            val = float('nan')
        else:
            true_labels = state.mapping_input.assignment.neuron_core_pre_assignment
            predicted_labels = state.c
            ari = adjusted_rand_score(true_labels, predicted_labels)
            val = float(ari)

        diff = val if baseline is not None else None
        return ObjectiveResult(self.get_name(), val, diff)

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

    def evaluate(self, state: BaseMosaicMappingState[Any], baseline: Optional[Any] = None) -> ObjectiveResult:
        if state is None or state.mapping_input is None or state.mapping_input.assignment is None:
            val = float('nan')
        else:
            true_labels = state.mapping_input.assignment.neuron_core_pre_assignment
            predicted_labels = state.c
            cm = confusion_matrix(true_labels, predicted_labels)
            # Use the Hungarian algorithm to find optimal 1-to-1 mapping
            row_ind, col_ind = linear_sum_assignment(-cm) # We pass -cm because the algorithm minimizes cost, but we want to maximize overlap
            accuracy = cm[row_ind, col_ind].sum() / np.size(true_labels)
            val = float(accuracy)

        diff = val if baseline is not None else None
        return ObjectiveResult(self.get_name(), val, diff)

    def get_name(self) -> str:
        return NAME_OBJ_CLUSTERING_ACCURACY


class CoreDistanceDifferenceObjective(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]):
    """
    Objective that measures the difference in hierarchical core distance
    between the true neuron assignments and the predicted neuron assignments.
    Calculates the mean absolute difference of pairwise distances.
    """

    @icontract.require(lambda config: isinstance(config, CoreDistanceDifferenceObjectiveConfig))
    def __init__(self, config: CoreDistanceDifferenceObjectiveConfig) -> None:
        super().__init__()
        self.config = config

    def evaluate(self, state: BaseMosaicMappingState[Any], baseline: Optional[Any] = None) -> ObjectiveResult:
        if state is None or state.mapping_input is None or state.mapping_input.assignment is None:
            val = float('nan')
        else:
            true_labels = state.mapping_input.assignment.neuron_core_pre_assignment
            predicted_labels = state.c

            n_neurons = len(true_labels)
            if n_neurons < 2:
                val = 0.0
            else:
                total_diff = 0.0
                count = 0

                # Calculate the mean absolute difference in pairwise core distances
                for i in range(n_neurons):
                    for j in range(i + 1, n_neurons):
                        dist_true = state.hw_to_evaluate.core_distance(true_labels[i], true_labels[j])
                        dist_pred = state.hw_to_evaluate.core_distance(predicted_labels[i], predicted_labels[j])
                        total_diff += abs(dist_true - dist_pred)
                        count += 1

                val = float(total_diff / count) if count > 0 else 0.0

        diff = val if baseline is not None else None
        return ObjectiveResult(self.get_name(), val, diff)

    def get_name(self) -> str:
        return NAME_OBJ_CORE_DISTANCE_DIFF