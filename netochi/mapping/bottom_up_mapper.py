from collections import defaultdict
from typing import List

import icontract
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from netochi.input_generator.interfaces import MosaicMappingInput, MosaicAssignment
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.mapping.three_step_mapping.clustering.clusterer.bottom_up_clusterer import build_new_temp_graph, \
    infer_core_assignment
from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import kaMinParClusterFunction
from netochi.mapping.three_step_mapping.interfaces import ClusterAndHwOutput
from netochi.mapping.three_step_mapping.local_address_assignment.top_down_local_address_assigner import SliceSolveMode, \
    ConstraintGraphMode, TopDownLocalAddressAssignerConfig
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner


class BottomUpMapperConfig(BaseModel):
    slice_solve_mode: SliceSolveMode = Field(
        default=SliceSolveMode.KL_CONSTRAINED_BISECTION,
        description="The algorithm used to solve the slice partitioning problem",
    )
    constraint_graph_mode: ConstraintGraphMode = Field(
        default=ConstraintGraphMode.CLIQUE,
        description="The strategy used to connect common sources in constraint graph",
    )

    def create(self) -> "BottomUpMapper":
        return BottomUpMapper(config=self)


class BottomUpMapper(BaseMapper[MosaicNetworkMappingState, MosaicMappingInput]):
    """
    Mapper that uses a bottom-up dual-hierarchy approach.
    Step 1: Groups neurons into Cores (randomly for this prototype).
    Step 2: Recursively splits slices globally using strictly balanced ILP Bisection.
    """

    @icontract.require(lambda config: isinstance(config, BottomUpMapperConfig))
    def __init__(self, config: BottomUpMapperConfig) -> None:
        self.config = config
        self.local_address_assigner = TopDownLocalAddressAssignerConfig(slice_solve_mode=config.slice_solve_mode, constraint_graph_mode=config.constraint_graph_mode).create()

    def run_mapping(self, mapping_input: MosaicMappingInput) -> MosaicNetworkMappingState:

        hw = mapping_input.hw_config
        graph = mapping_input.graph

        # ======== cluster nodes into cores and slices ===========
        temp_graph = graph
        hierarchy_mappings: List[npt.NDArray[np.int_]] = []
        local_address_bit_string: np.ndarray = np.zeros(graph.num_vertices(), dtype=int)
        original_vertex_to_current_cluster: np.ndarray = np.arange(graph.num_vertices(), dtype=int)


        for d in range(hw.router_levels + 1):
            print(f"  [Log] reached distance {d}/{hw.router_levels}")

            # Cluster temp graph
            cluster_size = mapping_input.hw_config.neurons_per_core if d == 0 else mapping_input.hw_config.nodes_per_router
            nr_clusters = int(temp_graph.num_vertices() / cluster_size)
            temp_clustering = kaMinParClusterFunction(graph=temp_graph, num_clusters=nr_clusters)

            # Map each old temp vertex ID to its new cluster ID
            temp_vertex_to_cluster: npt.NDArray[np.int_] = np.zeros(temp_graph.num_vertices(), dtype=int)
            for cluster_id, nodes in enumerate(temp_clustering):
                for v_id in nodes:
                    temp_vertex_to_cluster[int(v_id)] = cluster_id
            hierarchy_mappings.append(temp_vertex_to_cluster)

            # Map each original vertex ID to its new cluster ID
            for global_id in range(graph.num_vertices()):
                old_temp_node = original_vertex_to_current_cluster[global_id]
                original_vertex_to_current_cluster[global_id] = temp_vertex_to_cluster[old_temp_node]

            # save the base core assignment
            if d == 0:
                base_core_assignment = original_vertex_to_current_cluster.copy()

            # update the local address bit string: bisect slices
            if d < hw.router_levels:
                # gt_local_idx = mapping_input.assignment.neuron_idx_pre_assignment
                # total_bits = hw.router_levels
                temp_slice_split_assignment: np.ndarray = self.local_address_assigner.split_slice_clusters(local_address_bit_string=local_address_bit_string,
                                                                                      graph=graph,
                                                                                      current_cluster_assignment=original_vertex_to_current_cluster,
                                                                                      base_core_assignment=base_core_assignment,
                                                                                      seed=mapping_input.seed
                                                                                    )
                local_address_bit_string = (local_address_bit_string << 1) | temp_slice_split_assignment.astype(int)

            # build new temp_graph for the next hierarchy level
            temp_graph = build_new_temp_graph(temp_graph=temp_graph, nr_new_vertices=len(temp_clustering), cluster_assignment=temp_vertex_to_cluster)

        # ======== infer core assignment ============
        core_assignment: np.ndarray = infer_core_assignment(hierarchy_mappings=hierarchy_mappings)

        # ======== infer local assignment ===========
        local_assignment = np.zeros(graph.num_vertices(), dtype=int)

        # Group all neurons by their assigned core
        core_to_neurons = defaultdict(list)
        for neuron_id in range(graph.num_vertices()):
            core_id = core_assignment[neuron_id]
            core_to_neurons[core_id].append(neuron_id)

        # Assign contiguous local indices per core based on the slice bit string: ensures that every neuron receives unique local index within core even when 2^dmax < N
        for core_id, neurons_in_core in core_to_neurons.items():
            # Sort neurons in this core: Primarily by their slice ID, Secondarily by their neuron_id (for deterministic results)
            neurons_in_core.sort(key=lambda n: (local_address_bit_string[n], n))

            # Map them sequentially (Example: If N=8, Slice 0 gets 0,1,2,3 and Slice 1 gets 4,5,6,7)
            for local_idx, neuron_id in enumerate(neurons_in_core):
                local_assignment[neuron_id] = local_idx

        # ======== convert into mapping state =======
        clustering = ClusterAndHwOutput(cluster_assignment=core_assignment, num_clusters=mapping_input.hw_config.total_cores, hw=mapping_input.hw_config)
        slice_assignment = OptimalSliceAssigner().assign_slices(clustering, graph=graph, local_assignment=local_assignment)
        assignment = MosaicAssignment(
            hw=mapping_input.hw_config,
            neuron_core_pre_assignment=core_assignment,
            neuron_idx_pre_assignment=local_assignment,
            neuron_slice_assignment=slice_assignment
        )
        return MosaicNetworkMappingState(assignment=assignment, mapping_input=mapping_input)


    def get_name(self) -> str:
        return f"BottomUp_{self.config.constraint_graph_mode.name}_{self.config.slice_solve_mode.name}"