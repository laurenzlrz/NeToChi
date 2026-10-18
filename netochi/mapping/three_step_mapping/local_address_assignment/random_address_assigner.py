"""
Random address assigner (LocalAddressAssigner): gives each neuron a uniformly random,
unique local index within its core. Baseline for comparing index initializations.

Method: for every core, draw as many distinct indices from {0, ..., neurons_per_core - 1}
as the core holds neurons (a random subset in random order). If a core is not full,
the free slots are spread uniformly at random.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ----------------------------------------------------------------------------
# ADAPT these imports to your project layout
# ----------------------------------------------------------------------------
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import LocalAddressAssigner, ClusterAndHwOutput  # ADAPT


class RandomAddressAssigner(LocalAddressAssigner):
    """Uniformly random, unique local index per core."""

    def assign_addresses(self, mapping_input: MosaicMappingInput,
                         clustering: ClusterAndHwOutput) -> npt.NDArray[np.int_]:
        """neuron_id -> local index within its core (unique within each core)."""
        hw = clustering.hw
        core = np.asarray(clustering.cluster_assignment, dtype=np.int64)
        per_core = hw.neurons_per_core

        rng = np.random.default_rng(mapping_input.seed)
        idx = np.zeros(len(core), dtype=np.int64)
        for c in range(hw.total_cores):
            nodes = np.flatnonzero(core == c)
            if nodes.size == 0:
                continue
            idx[nodes] = rng.choice(per_core, size=nodes.size, replace=False)
        return idx
