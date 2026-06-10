from typing import Any

import numpy as np
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig


class SAState:

    def __init__(self, num_nodes: int, hw_config: MosaicHardwareConfig):
        self.hw_config = hw_config

        num_slots = hw_config.total_cores * hw_config.neurons_per_core
        if num_slots < num_nodes:
            raise ValueError("Not enough hardware slots for the input graph nodes.")

        # --- random initial assignment ---
        # initial_flat_slots = np.arange(num_nodes) # TODO use this initialization so that the mapper passes the test
        initial_flat_slots = np.random.choice(num_slots, size=num_nodes, replace=False)
        self.core_assignment: np.ndarray[int] = initial_flat_slots // hw_config.neurons_per_core
        self.local_assignment: np.ndarray[int] = initial_flat_slots % hw_config.neurons_per_core

        self.slot_to_node: np.ndarray[tuple[Any, Any], np.dtype[np.int_]] = np.full((hw_config.total_cores, hw_config.neurons_per_core), -1, dtype=np.int_)
        self.slot_to_node[self.core_assignment, self.local_assignment] = np.arange(num_nodes)

