from typing import List

import pytest

from netochi.input_generator.erdos_renyi_factory import ErdosRenyiConfig
from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
from netochi.mapping.greedy_mapper import GreedyMapperConfig
from netochi.mapping.interfaces import BaseMosaicMappingState
from netochi.mapping.three_step_mapping.clustering.qap_hw_clusterer import QapHwClusterer
from netochi.mapping.three_step_mapping.local_address_assignment.pca_local_address_assigner import \
    PcaLocalAddressAssigner
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner
from netochi.mapping.three_step_mapping.three_step_hw_mapper import ThreeStepHwMapperConfig
from netochi.objectives.obj_inconsistency import InconsistencyObjectiveConfig
from netochi.objectives.obj_unused_connections import UnusedConnectionsObjectiveConfig
from netochi.objectives.obj_valid_edges import ValidEdgesObjectiveConfig
from netochi.objectives.utils import compute_total_hw_connections


@pytest.fixture
def hardware_config():
    return MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=4,
        router_levels=2,
        slice_factor=2
    )


def inconsistency_objective():
    return InconsistencyObjectiveConfig().create()

def unused_connections_objective():
    return UnusedConnectionsObjectiveConfig().create()

def valid_edges_objective():
    return ValidEdgesObjectiveConfig().create()



def verify_equations(state: BaseMosaicMappingState[MosaicMappingInput]):
    """Helper to assert the two core equations."""
    num_edges = state.mapping_input.graph.num_edges()
    num_edges_on_chip = state.hw_to_evaluate.total_hardware_edges
    num_edges_on_chip_2 = compute_total_hw_connections(state.hw_to_evaluate)

    assert num_edges_on_chip == num_edges_on_chip_2, "num_edges_on_chip do not match!"

    e_valid = valid_edges_objective().evaluate(state).value
    inconsistencies = inconsistency_objective().evaluate(state).value
    unused_connections = unused_connections_objective().evaluate(state).value


    # Equation 1: Every edge is either valid or inconsistent
    assert num_edges == e_valid + inconsistencies, \
        f"Eq 1 Failed: {num_edges} logical edges != {e_valid} valid + {inconsistencies} inconsistent"

    # Equation 2: Unused connections represent leftover capacity
    assert num_edges_on_chip == unused_connections + e_valid, \
        f"Eq 2 Failed: Unused mismatch. Total HW: {num_edges_on_chip}, Used: {e_valid}"


def test_for_random_mappings(hardware_config):
    # generate random graph with factories
    # use two different mappers
    # check whether output fulfills constraint
    SEED = 42

    mosaic_factory = MosaicNetworkConfig(hw_config=hardware_config, probability=0.9, seed=SEED).create()
    er_factory = ErdosRenyiConfig(hw_config=hardware_config, n=hardware_config.total_neurons, probability=0.3, seed=SEED).create()
    factories: List[HWBaseInputFactory[MosaicMappingInput]] = [mosaic_factory, er_factory]

    mappers = [
        ThreeStepHwMapperConfig(clusterer=QapHwClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
        GreedyMapperConfig().create(),
    ]

    for mapper in mappers:
        for factory in factories:
            mapping_input = factory.generate()
            state = mapper.run(mapping_input=mapping_input)
            verify_equations(state)

