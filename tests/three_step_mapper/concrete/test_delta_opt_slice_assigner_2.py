import pytest
import numpy as np
import random

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.simulated_annealing_fix_hw.sa_mutation import SwapMutation, MoveMutation
from netochi.mapping.simulated_annealing_fix_hw.sa_state import SAState
from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner


# ==========================================
# MOCK DEPENDENCIES
# ==========================================

class MockClusterAndHwOutput:
    def __init__(self, hw, cluster_assignment):
        self.hw = hw
        self.cluster_assignment = cluster_assignment



# ==========================================
# TEST CLASS
# ==========================================
import pytest
import random
import numpy as np
import graph_tool.all as gt

SWAP = 0
MOVE = 1


class TestDeltaVsGroundTruth:

    @pytest.fixture
    def setup_large_graph(self):
        """
        Creates a random graph and a fully populated SAState with slot_to_node tracking.
        """
        np.random.seed(42)
        random.seed(42)

        hw_config = MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=8, router_levels=2)
        num_nodes = hw_config.total_neurons // 2  # Keep hardware half-empty to allow Move mutations
        state = SAState(num_nodes, hw_config)

        # --- generate random graph ---
        graph = gt.Graph(directed=True)
        graph.add_vertex(num_nodes)

        # generate ~50 random edges
        for i in range(50):
            src = random.randint(0, num_nodes - 1)
            tgt = random.randint(0, num_nodes - 1)
            if src != tgt:
                graph.add_edge(src, tgt)

        return hw_config, graph, state

    def assert_assignments_match(self, delta_assigner, ground_truth_assigner, state, graph):
        """
        Helper method to assert delta state matches ground truth state.
        """
        clustering = MockClusterAndHwOutput(state.hw_config, state.core_assignment)
        gt_assignment = ground_truth_assigner.assign_slices(clustering, graph, state.local_assignment)
        delta_assignment = delta_assigner.slice_assignment

        np.testing.assert_array_equal(
            delta_assignment,
            gt_assignment,
            err_msg="Delta assigner output drifted from ground truth assigner!"
        )

    def generate_random_mutation(self, state, delta_assigner, graph):
        """
        Helper to pick and execute a random Swap or Move mutation.
        """
        num_nodes = len(state.core_assignment)
        num_slots = state.slot_to_node.size

        # Decide between a Swap (0) or a Move to an empty slot (1)
        move_type = random.choice([SWAP, MOVE]) if num_slots > num_nodes else SWAP

        if move_type == SWAP:
            node_a, node_b = random.sample(range(num_nodes), 2)
            mutation = SwapMutation(node_a, node_b)
        else:
            node = random.randint(0, num_nodes - 1)
            empty_cores, empty_locals = np.where(state.slot_to_node == -1)
            random_idx = random.randrange(len(empty_cores))
            mutation = MoveMutation(node, empty_cores[random_idx].item(), empty_locals[random_idx].item())

        mutation.do(state=state, slice_assigner=delta_assigner, graph=graph)
        return mutation

    def test_random_mutations_vs_ground_truth(self, setup_large_graph):
        hw_config, graph, state = setup_large_graph

        delta_assigner = DeltaOptimalSliceAssigner(hw_config, graph, state.core_assignment, state.local_assignment)
        gt_assigner = OptimalSliceAssigner()

        # Perform 10 sequential mutations (mix of moves and swaps)
        for _ in range(10):
            self.generate_random_mutation(state, delta_assigner, graph)
            self.assert_assignments_match(delta_assigner, gt_assigner, state, graph)

    def test_undo_mutations_vs_ground_truth(self, setup_large_graph):
        hw_config, graph, state = setup_large_graph

        delta_assigner = DeltaOptimalSliceAssigner(hw_config, graph, state.core_assignment, state.local_assignment)
        gt_assigner = OptimalSliceAssigner()

        # Baseline ground truth
        baseline_gt = gt_assigner.assign_slices(
            MockClusterAndHwOutput(hw_config, state.core_assignment),
            graph,
            state.local_assignment
        )

        for _ in range(5):
            # Apply mutation
            mutation = self.generate_random_mutation(state, delta_assigner, graph)

            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(delta_assigner.slice_assignment, delta_assigner.backup_slice_assignment, err_msg="assignment did not change")

            # Undo mutation
            mutation.undo(state=state, slice_assigner=delta_assigner)

            # Assert state reverted perfectly to the baseline
            np.testing.assert_array_equal(
                delta_assigner.slice_assignment,
                baseline_gt,
                err_msg="Delta assigner failed to correctly undo mutation state!"
            )

    def test_random_swaps_vs_ground_truth(self, setup_large_graph):
        hw_config, graph, state = setup_large_graph

        # 1. Initialize both assigners using the current framework state
        delta_assigner = DeltaOptimalSliceAssigner(hw_config, graph, state.core_assignment, state.local_assignment)
        gt_assigner = OptimalSliceAssigner()

        num_nodes = len(state.core_assignment)

        # Perform sequential rounds of swaps using the Mutation classes
        for _ in range(5):
            # Select 2 unique nodes to swap
            node_a, node_b = random.sample(range(num_nodes), 2)

            # Instantiate and apply the framework mutation
            swap_mutation = SwapMutation(node_a, node_b)
            swap_mutation.do(state=state, slice_assigner=delta_assigner, graph=graph)

            # Compare to Ground Truth using the framework's verification method
            self.assert_assignments_match(delta_assigner, gt_assigner, state, graph)

    def test_undo_mechanics_vs_ground_truth(self, setup_large_graph):
        """
        Tests that mutation.undo() correctly reverts both the SAState
        and the delta slice assigner back to the exact baseline ground truth.
        """
        hw_config, graph, state = setup_large_graph

        delta_assigner = DeltaOptimalSliceAssigner(hw_config, graph, state.core_assignment, state.local_assignment)
        gt_assigner = OptimalSliceAssigner()

        # Gather baseline ground truth slice assignment before applying mutations
        clustering = MockClusterAndHwOutput(state.hw_config, state.core_assignment)
        baseline_gt = gt_assigner.assign_slices(clustering, graph, state.local_assignment)

        num_nodes = len(state.core_assignment)

        # Pick two unique nodes and execute a Swap mutation
        node_a, node_b = random.sample(range(num_nodes), 2)
        mutation = SwapMutation(node_a, node_b)
        mutation.do(state=state, slice_assigner=delta_assigner, graph=graph)

        # Undo the mutation using the framework abstraction
        mutation.undo(state=state, slice_assigner=delta_assigner)

        # Verify that after the undo, the delta assigner matches the BASELINE ground truth
        np.testing.assert_array_equal(
            delta_assigner.slice_assignment,
            baseline_gt,
            err_msg="Delta assigner failed to correctly restore state after a mutation undo!"
        )