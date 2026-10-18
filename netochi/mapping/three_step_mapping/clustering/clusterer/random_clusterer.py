"""
Random clusterer (ClustererFixedHw): assigns neurons to cores uniformly at random,
respecting core capacities. Baseline for comparing core initializations.

Method: shuffle all hardware slots (core, local index) and give the first |V| of
them to the neurons. Every injective placement is equally likely, so each core
holds at most neurons_per_core neurons, and if the graph fills the hardware, every
core is full.

Output convention (same as SpectralTreeClusterer): cluster id == core id and
num_clusters == hw.total_cores (some clusters may be empty).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# ADAPT these imports to your project layout
# ----------------------------------------------------------------------------
from netochi.input_generator.interfaces import MosaicMappingInput
from netochi.mapping.three_step_mapping.interfaces import ClustererFixedHw, ClusterAndHwOutput   # ADAPT


class RandomClusterer(ClustererFixedHw[MosaicMappingInput]):
    """Uniformly random core assignment that respects core capacities."""

    def cluster(self, input_data: MosaicMappingInput) -> ClusterAndHwOutput:
        hw = input_data.hw_config
        n = int(input_data.graph.num_vertices())
        n_cores, per_core = hw.total_cores, hw.neurons_per_core

        rng = np.random.default_rng(input_data.seed)
        # one entry per hardware slot, labelled with its core; a random subset of size n
        slots = np.repeat(np.arange(n_cores, dtype=np.int64), per_core)
        core = rng.permutation(slots)[:n]

        return ClusterAndHwOutput(cluster_assignment=core, num_clusters=n_cores, hw=hw)
