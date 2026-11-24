"""
!!!! for this graph we have nr cores = nr excitatory clusters !!!!
Structure-aware mapper for sWTA v2 networks (see ``SwtaV2Factory``): one E cluster per core plus
one global inhibitory cluster of size total_cores * q.

Construction (no cluster is split):
1. E cluster k goes completely onto core k. Cores are numbered in the order of the router tree,
   so consecutive chain clusters are as close as possible (E_k - E_{k+1} at distance 1 + trailing
   zeros of k + 1).
2. The inhibitory cluster is spread evenly: q neurons per core. With q <= neurons_per_core / 2^L
   they all fit into slice 0 at every distance, so every target sees every inhibitory neuron.
3. Local indices are solved exactly per core under the slice-0 policy (see ``swta_mapper``), so
   the low, far-visible indices go to the neurons with the most remote out-edges (I first).
4. The slice per (target, distance) is then chosen exactly.

The returned mapping is valid, so its inconsistency count is an upper bound on the optimum.
"""
from typing import Optional, Tuple

import icontract
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from netochi.input_generator.interfaces import MosaicAssignment, MosaicMappingInput
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.mapping.swta_mapper import _LayoutEvaluator, optimal_slices

# Description keys written by SwtaV2Factory
DESC_NUM_E_CLUSTERS = "num_e_clusters"
DESC_E_CLUSTER_SIZE = "e_cluster_size"


class SwtaV2MapperConfig(BaseModel):
    num_e_clusters: Optional[int] = Field(default=None, gt=0, description="Number of E clusters. Read from the input descriptions if None.")
    e_cluster_size: Optional[int] = Field(default=None, gt=0, description="Neurons per E cluster. Read from the input descriptions if None.")

    def create(self) -> "SwtaV2Mapper":
        return SwtaV2Mapper(config=self)


class SwtaV2Mapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """Places every E cluster whole on its own core and spreads the inhibitory cluster evenly."""

    @icontract.require(lambda config: isinstance(config, SwtaV2MapperConfig))
    def __init__(self, config: SwtaV2MapperConfig) -> None:
        self.config = config

    def get_name(self) -> str:
        return "SwtaV2Mapper"

    def _structure(self, mapping_input: MosaicMappingInput) -> Tuple[int, int]:
        desc = mapping_input.descriptions

        def get(value: Optional[int], key: str) -> int:
            if value is not None:
                return value
            if key not in desc:
                raise ValueError(f"SwtaV2Mapper needs '{key}': set it in SwtaV2MapperConfig or in the input descriptions.")
            return int(desc[key])

        num_e = get(self.config.num_e_clusters, DESC_NUM_E_CLUSTERS)
        e_size = get(self.config.e_cluster_size, DESC_E_CLUSTER_SIZE)
        hw = mapping_input.hw_config
        if num_e != hw.total_cores:
            raise ValueError(f"SwtaV2Mapper expects one E cluster per core ({num_e} clusters, {hw.total_cores} cores).")
        num_i = mapping_input.num_neurons - num_e * e_size
        if num_i < 0 or e_size + -(-num_i // hw.total_cores) > hw.neurons_per_core:
            raise ValueError("E cluster plus its share of inhibitory neurons does not fit onto one core.")
        return num_e, e_size

    def layout(self, mapping_input: MosaicMappingInput) -> npt.NDArray[np.int64]:
        """Core per neuron: E cluster k on core k, inhibitory neurons round-robin over all cores."""
        num_e, e_size = self._structure(mapping_input)
        num_neurons = mapping_input.num_neurons
        cores = np.empty(num_neurons, dtype=np.int64)
        num_excitatory = num_e * e_size
        cores[:num_excitatory] = np.arange(num_excitatory) // e_size
        cores[num_excitatory:] = np.arange(num_neurons - num_excitatory) % mapping_input.hw_config.total_cores
        return cores

    def run_mapping(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        hw = mapping_input.hw_config
        edges = mapping_input.graph.get_edges()
        src = edges[:, 0].astype(np.int64)
        tgt = edges[:, 1].astype(np.int64)

        cores = self.layout(mapping_input)
        _, indices = _LayoutEvaluator(hw, src, tgt, mapping_input.num_neurons, cores).value()
        slices = optimal_slices(hw, src, tgt, cores, indices)

        assignment = MosaicAssignment(
            hw=hw,
            neuron_core_pre_assignment=cores,
            neuron_idx_pre_assignment=indices,
            neuron_slice_assignment=slices,
        )
        return MosaicNetworkMappingState(mapping_input=mapping_input, assignment=assignment)
