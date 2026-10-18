"""
Theoretical-bounds mapper for the Mosaic hardware.

Implements the recovery algorithm of the proof "Recovering a Planted Hierarchical Graph, by
Semidefinite Programming" (TheoreticalBoundsProof/alg1/RESULT_v2_readable.md, Chapters 2 and 6)
as a regular (not three-step) mapper:

  1. core stage       balanced-partition SDP on W = A + A^T, rounded to C blocks of exactly n neurons
  2. router tree      block statistic + recursive bisection -> blocks placed on cores
  3. prefix stage     per root half: co-occurrence SDP -> top-h address prefix classes
  4. address blocks   recursive bisection of Psi_d (summed over all reference cores) -> prefix labels
  5. slices           OptimalSliceAssigner (exact per (target, distance) argmax, Proposition 1)

Deviations from the proof's text (robustness, all configurable or documented):
  - the SDP solutions are rounded (balanced k-means on the rows of Y, exact balance per core);
    at exact recovery the rounding is the identity on the block structure,
  - block_statistic="median" (default) replaces the proof's block totals in step 2,
  - Psi_d is summed over all reference cores of the half instead of a single one,
  - each bisection takes the best of the own-row proposals of all items (Section 6.2's own-row rule)
    instead of a single reference block.

Supported hardware (anything else raises): nodes_per_router = 2, slice_factor = 2, router_levels >= 1,
neurons_per_core = 2^L with 2^L >= 2^h (enough address bits for the h-bit prefix), and the graph
must fill the hardware exactly (|V| = total_cores * neurons_per_core).

Scalability: the core SDP has N^2 variables (N = |V|) and an N x N PSD constraint; the two prefix
SDPs are (N/2) x (N/2). With SCS this is fine up to a few hundred neurons; `max_neurons` guards
against accidentally launching it on large instances.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import icontract
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from netochi.definitions.exceptions import HardwareConstraintError, MappingError, MappingValidationError
from netochi.input_generator.interfaces import MosaicAssignment, MosaicMappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.mapping.theoretical_bounds.cell_recovery import CellRecoveryResult, recover_cells
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput, SliceAssigner
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner

UNSUPPORTED_ROUTER_ARITY = "TheoreticalBoundsMapper needs binary routers (nodes_per_router = 2), got {value}."
UNSUPPORTED_SLICE_FACTOR = "TheoreticalBoundsMapper needs slice_factor = 2 (dyadic slices), got {value}."
UNSUPPORTED_ROUTER_LEVELS = "TheoreticalBoundsMapper needs router_levels >= 1, got {value}."
UNSUPPORTED_CORE_SIZE = "TheoreticalBoundsMapper needs neurons_per_core to be a power of two, got {value}."
TOO_FEW_ADDRESS_BITS = (
    "TheoreticalBoundsMapper needs neurons_per_core >= 2^router_levels (an h-bit address prefix), "
    "got neurons_per_core = {per_core} < 2^{levels}."
)
GRAPH_DOES_NOT_FILL_HW = (
    "TheoreticalBoundsMapper needs |V| = total_cores * neurons_per_core = {expected}, got |V| = {actual}."
)
TOO_MANY_NEURONS_FOR_SDP = (
    "TheoreticalBoundsMapper: {actual} neurons exceed max_neurons = {limit} (the core SDP has N^2 variables)."
)


class TheoreticalBoundsMapperConfig(BaseModel):
    """Configuration of the SDP-based TheoreticalBoundsMapper."""
    model_config = ConfigDict(frozen=True)

    block_statistic: Literal["total", "median"] = Field(
        default="median",
        description="Block-to-block statistic for the router tree: 'total' = sum_{u in a, x in b} W_ux "
                    "(proof, Section 6.1), 'median' = median_{u in a} W(u, b) (robust variant).",
    )
    solver: str = Field(default="SCS", description="cvxpy solver used for the SDPs.")
    solver_options: Dict[str, Union[int, float, bool, str]] = Field(
        default_factory=dict, description="Keyword arguments passed to cvxpy's Problem.solve."
    )
    rounding_iterations: int = Field(
        default=20, gt=0, description="Max. iterations of the balanced k-means rounding of each SDP solution."
    )
    max_neurons: int = Field(
        default=1024, gt=0, description="Refuse instances with more neurons (the core SDP has N^2 variables)."
    )

    def create(self) -> "TheoreticalBoundsMapper":
        return TheoreticalBoundsMapper(config=self)


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


class TheoreticalBoundsMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    SDP-based recovery mapper (cores, router tree, address prefixes, address blocks, slices).
    In the noise-free Mosaic model at sufficient density it recovers the planted cells exactly and
    hence produces zero inconsistencies (RESULT_v2_readable.md, Main Theorem).
    """

    @icontract.require(lambda config: isinstance(config, TheoreticalBoundsMapperConfig))
    def __init__(self, config: TheoreticalBoundsMapperConfig) -> None:
        self.config = config
        self.slice_assigner: SliceAssigner = OptimalSliceAssigner()
        self.last_recovery: Optional[CellRecoveryResult] = None

    def check_supported(self, hw: MosaicHardwareConfig, num_neurons: int) -> None:
        """Raise if the hardware / input is outside the model the algorithm is defined for."""
        if hw.nodes_per_router != 2:
            raise HardwareConstraintError(UNSUPPORTED_ROUTER_ARITY.format(value=hw.nodes_per_router))
        if hw.slice_factor != 2:
            raise HardwareConstraintError(UNSUPPORTED_SLICE_FACTOR.format(value=hw.slice_factor))
        if hw.router_levels < 1:
            raise HardwareConstraintError(UNSUPPORTED_ROUTER_LEVELS.format(value=hw.router_levels))
        if not _is_power_of_two(hw.neurons_per_core):
            raise HardwareConstraintError(UNSUPPORTED_CORE_SIZE.format(value=hw.neurons_per_core))
        if hw.neurons_per_core < 2 ** hw.router_levels:
            raise HardwareConstraintError(
                TOO_FEW_ADDRESS_BITS.format(per_core=hw.neurons_per_core, levels=hw.router_levels)
            )
        if num_neurons != hw.total_neurons:
            raise MappingValidationError(GRAPH_DOES_NOT_FILL_HW.format(expected=hw.total_neurons, actual=num_neurons))
        if num_neurons > self.config.max_neurons:
            raise MappingError(TOO_MANY_NEURONS_FOR_SDP.format(actual=num_neurons, limit=self.config.max_neurons))

    @icontract.ensure(lambda result, mapping_input: result.c.size == mapping_input.graph.num_vertices())
    def run_mapping(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:
        """Recover cells by the two SDP stages and the two tree bisections, then fit the slices."""
        hw: MosaicHardwareConfig = mapping_input.hw_config
        graph = mapping_input.graph
        num_neurons = int(graph.num_vertices())
        self.check_supported(hw, num_neurons)

        edges: npt.NDArray[np.int64] = np.asarray(graph.get_edges(), dtype=np.int64).reshape(-1, 2)[:, :2]
        adjacency: npt.NDArray[np.float64] = np.zeros((num_neurons, num_neurons), dtype=np.float64)
        adjacency[edges[:, 0], edges[:, 1]] = 1.0
        np.fill_diagonal(adjacency, 0.0)

        recovery = recover_cells(
            adjacency,
            hw,
            rng=np.random.default_rng(mapping_input.seed),
            block_statistic=self.config.block_statistic,
            solver=self.config.solver,
            solver_options=self.config.solver_options,
            rounding_iterations=self.config.rounding_iterations,
        )
        self.last_recovery = recovery

        clustering = ClusterAndHwOutput(cluster_assignment=recovery.core, num_clusters=hw.total_cores, hw=hw)
        slices = self.slice_assigner.assign_slices(
            clustering=clustering, graph=graph, local_assignment=recovery.local_address
        )

        assignment = MosaicAssignment(
            hw=hw,
            neuron_core_pre_assignment=recovery.core.astype(np.int64),
            neuron_idx_pre_assignment=recovery.local_address.astype(np.int64),
            neuron_slice_assignment=np.asarray(slices, dtype=np.int64),
        )
        return MosaicNetworkMappingState(assignment=assignment, mapping_input=mapping_input)
