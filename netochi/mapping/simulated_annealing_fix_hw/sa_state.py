from typing import Any

import numpy as np

from netochi.input_generator.interfaces import MosaicHWMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from tests.test_validate_mappers_general import hw_config


class SAState:

    def __init__(self, mapping_input: MosaicHWMappingInput):
        hw_config = mapping_input.hw_config
        num_nodes = mapping_input.graph.num_vertices()
        self.hw_config = hw_config

        num_slots = hw_config.total_cores * hw_config.neurons_per_core
        if num_slots < num_nodes:
            raise ValueError("Not enough hardware slots for the input graph nodes.")

        # --- random initial assignment ---
        # initial_flat_slots = np.arange(num_nodes) # TODO use this initialization so that the mapper passes the test
        if mapping_input.core_assignment_initialization is None:
            initial_flat_slots = np.random.choice(num_slots, size=num_nodes, replace=False)
            self.core_assignment: np.ndarray[int] = initial_flat_slots // hw_config.neurons_per_core
            self.local_assignment: np.ndarray[int] = initial_flat_slots % hw_config.neurons_per_core
        else:
            core_assignment = mapping_input.core_assignment_initialization
            self.core_assignment = np.asarray(core_assignment, dtype=np.int_)
            self.local_assignment = np.zeros(num_nodes, dtype=np.int_)
            next_local = np.zeros(hw_config.total_cores, dtype=np.int_)
            for node, core in enumerate(self.core_assignment):
                local = next_local[core]
                self.local_assignment[node] = local
                next_local[core] += 1


        self.slot_to_node: np.ndarray[tuple[Any, Any], np.dtype[np.int_]] = np.full((hw_config.total_cores, hw_config.neurons_per_core), -1, dtype=np.int_)
        self.slot_to_node[self.core_assignment, self.local_assignment] = np.arange(num_nodes)

