from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import graph_tool.all as gt
import math

from netochi.input_generator.interfaces import MosaicAssignment
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.mapping.simulated_annealing_inferred_hw.sa_ihw_state import SAIHWState
from netochi.mapping.three_step_mapping.slice_assignment.delta_optimal_slice_assigner import DeltaOptimalSliceAssigner


class IHWMutation:
    """Base class for Inferred Hardware Mutations."""
    
    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        raise NotImplementedError()

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        raise NotImplementedError()

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWMutation']:
        raise NotImplementedError()


class IHWSwapMutation(IHWMutation):
    def __init__(self, node_a: int, node_b: int):
        self.node_a = node_a
        self.node_b = node_b

    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        core_a, local_a = int(state.c[self.node_a]), int(state.x[self.node_a])
        core_b, local_b = int(state.c[self.node_b]), int(state.x[self.node_b])
        
        state.c[self.node_a], state.x[self.node_a] = core_b, local_b
        state.c[self.node_b], state.x[self.node_b] = core_a, local_a
        
        slice_assigner.delta_assign_slices(
            state=state,
            moved_nodes=[self.node_a, self.node_b],
            graph=graph,
            old_core_and_local_of_moved_nodes={
                self.node_a: (core_a, local_a),
                self.node_b: (core_b, local_b)
            }
        )
        return slice_assigner

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        core_a, local_a = int(state.c[self.node_a]), int(state.x[self.node_a])
        core_b, local_b = int(state.c[self.node_b]), int(state.x[self.node_b])
        
        state.c[self.node_a], state.x[self.node_a] = core_b, local_b
        state.c[self.node_b], state.x[self.node_b] = core_a, local_a
        
        slice_assigner.undo_assign_slices()
        return slice_assigner

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWSwapMutation']:
        num_nodes = len(state.c)
        if num_nodes < 2:
            return None
        node_a, node_b = rng.choice(num_nodes, size=2, replace=False)
        return cls(int(node_a), int(node_b))


class IHWMoveMutation(IHWMutation):
    def __init__(self, node: int, new_core: int, new_local: int):
        self.node = node
        self.new_core = new_core
        self.new_local = new_local
        self.old_core: int = -1
        self.old_local: int = -1

    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        self.old_core = int(state.c[self.node])
        self.old_local = int(state.x[self.node])
        
        state.c[self.node] = self.new_core
        state.x[self.node] = self.new_local
        
        slice_assigner.delta_assign_slices(
            state=state,
            moved_nodes=[self.node],
            graph=graph,
            old_core_and_local_of_moved_nodes={
                self.node: (self.old_core, self.old_local)
            }
        )
        return slice_assigner

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        state.c[self.node] = self.old_core
        state.x[self.node] = self.old_local
        
        slice_assigner.undo_assign_slices()
        return slice_assigner

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWMoveMutation']:
        num_nodes = len(state.c)
        node = int(rng.integers(0, num_nodes))
        
        occupied = set(zip(state.c, state.x))
        num_slots = state.K * state.Nc
        if num_slots <= num_nodes:
            return None
        
        for _ in range(100):
            c = int(rng.integers(0, state.K))
            x = int(rng.integers(0, state.Nc))
            if (c, x) not in occupied:
                return cls(node, c, x)
        return None


class IHWAddCoreMutation(IHWMutation):
    def __init__(self, node: int):
        self.node = node
        self.old_hw: Optional[MosaicHardwareConfig] = None
        self.old_assignment: Optional[MosaicAssignment] = None
        self.old_assigner: Optional[DeltaOptimalSliceAssigner] = None

    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        self.old_hw = state.hw_config
        self.old_assignment = state.assignment
        self.old_assigner = slice_assigner

        new_K = state.K + 1
        new_L = max(1, int(math.ceil(math.log(new_K) / math.log(state.Nr))))
        new_hw = MosaicHardwareConfig(
            nodes_per_router=state.Nr,
            neurons_per_core=state.Nc,
            router_levels=new_L,
            slice_factor=state.slice_factor
        )

        new_c = state.c.copy()
        new_x = state.x.copy()
        new_c[self.node] = state.K
        new_x[self.node] = 0

        new_slices = np.zeros((len(state.c), new_L + 1), dtype=np.int64)
        min_L = min(self.old_hw.router_levels, new_L)
        new_slices[:, :min_L + 1] = state.s[:, :min_L + 1]

        new_assignment = MosaicAssignment(
            hw=new_hw,
            neuron_core_pre_assignment=new_c,
            neuron_idx_pre_assignment=new_x,
            neuron_slice_assignment=new_slices
        )
        state.update_assignment(new_assignment)

        new_assigner = DeltaOptimalSliceAssigner(
            hw_config=new_hw,
            graph=graph,
            cluster_assignment=state.c,
            local_assignment=state.x
        )
        state._slice_assigner = new_assigner
        return new_assigner

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        assert self.old_assignment is not None
        assert self.old_assigner is not None
        state.update_assignment(self.old_assignment)
        state._slice_assigner = self.old_assigner
        return self.old_assigner

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWAddCoreMutation']:
        num_nodes = len(state.c)
        node = int(rng.integers(0, num_nodes))
        return cls(node)


class IHWRemoveCoreMutation(IHWMutation):
    def __init__(self, core_to_remove: int):
        self.core_to_remove = core_to_remove
        self.old_assignment: Optional[MosaicAssignment] = None
        self.old_assigner: Optional[DeltaOptimalSliceAssigner] = None

    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        self.old_assignment = state.assignment
        self.old_assigner = slice_assigner

        new_c = state.c.copy()
        new_x = state.x.copy()
        new_c[new_c > self.core_to_remove] -= 1

        new_K = state.K - 1
        new_L = max(1, int(math.ceil(math.log(new_K) / math.log(state.Nr))))
        new_hw = MosaicHardwareConfig(
            nodes_per_router=state.Nr,
            neurons_per_core=state.Nc,
            router_levels=new_L,
            slice_factor=state.slice_factor
        )

        new_assignment = MosaicAssignment(
            hw=new_hw,
            neuron_core_pre_assignment=new_c,
            neuron_idx_pre_assignment=new_x,
            neuron_slice_assignment=state.s[:, :new_L + 1]
        )
        state.update_assignment(new_assignment)

        new_assigner = DeltaOptimalSliceAssigner(
            hw_config=new_hw,
            graph=graph,
            cluster_assignment=state.c,
            local_assignment=state.x
        )
        state._slice_assigner = new_assigner
        return new_assigner

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        assert self.old_assignment is not None
        assert self.old_assigner is not None
        state.update_assignment(self.old_assignment)
        state._slice_assigner = self.old_assigner
        return self.old_assigner

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWRemoveCoreMutation']:
        if state.K <= 1:
            return None
        used_cores = set(state.c)
        empty_cores = [c for c in range(state.K) if c not in used_cores]
        if not empty_cores:
            return None
        core_to_remove = int(rng.choice(empty_cores))
        return cls(core_to_remove)


class IHWIncrementNcMutation(IHWMutation):
    def __init__(self, node: int, target_core: int):
        self.node = node
        self.target_core = target_core
        self.old_assignment: Optional[MosaicAssignment] = None
        self.old_assigner: Optional[DeltaOptimalSliceAssigner] = None

    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        self.old_assignment = state.assignment
        self.old_assigner = slice_assigner

        new_Nc = state.Nc + 1
        new_hw = MosaicHardwareConfig(
            nodes_per_router=state.Nr,
            neurons_per_core=new_Nc,
            router_levels=state.L,
            slice_factor=state.slice_factor
        )

        new_c = state.c.copy()
        new_x = state.x.copy()
        new_c[self.node] = self.target_core
        new_x[self.node] = state.Nc

        new_assignment = MosaicAssignment(
            hw=new_hw,
            neuron_core_pre_assignment=new_c,
            neuron_idx_pre_assignment=new_x,
            neuron_slice_assignment=state.s.copy()
        )
        state.update_assignment(new_assignment)

        new_assigner = DeltaOptimalSliceAssigner(
            hw_config=new_hw,
            graph=graph,
            cluster_assignment=state.c,
            local_assignment=state.x
        )
        state._slice_assigner = new_assigner
        return new_assigner

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        assert self.old_assignment is not None
        assert self.old_assigner is not None
        state.update_assignment(self.old_assignment)
        state._slice_assigner = self.old_assigner
        return self.old_assigner

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWIncrementNcMutation']:
        num_nodes = len(state.c)
        core_counts = np.bincount(state.c, minlength=state.K)
        full_cores = np.where(core_counts >= state.Nc)[0]
        if len(full_cores) == 0:
            return None
        target_core = int(rng.choice(full_cores))
        node = int(rng.integers(0, num_nodes))
        return cls(node, target_core)


class IHWDecrementNcMutation(IHWMutation):
    def __init__(self, evicted_swaps: List[Tuple[int, Tuple[int, int]]]):
        self.evicted_swaps = evicted_swaps
        self.old_assignment: Optional[MosaicAssignment] = None
        self.old_assigner: Optional[DeltaOptimalSliceAssigner] = None

    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        self.old_assignment = state.assignment
        self.old_assigner = slice_assigner

        new_c = state.c.copy()
        new_x = state.x.copy()

        for real_u, (c, x) in self.evicted_swaps:
            new_c[real_u] = c
            new_x[real_u] = x

        new_Nc = state.Nc - 1
        new_hw = MosaicHardwareConfig(
            nodes_per_router=state.Nr,
            neurons_per_core=new_Nc,
            router_levels=state.L,
            slice_factor=state.slice_factor
        )

        new_assignment = MosaicAssignment(
            hw=new_hw,
            neuron_core_pre_assignment=new_c,
            neuron_idx_pre_assignment=new_x,
            neuron_slice_assignment=state.s.copy()
        )
        state.update_assignment(new_assignment)

        new_assigner = DeltaOptimalSliceAssigner(
            hw_config=new_hw,
            graph=graph,
            cluster_assignment=state.c,
            local_assignment=state.x
        )
        state._slice_assigner = new_assigner
        return new_assigner

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        assert self.old_assignment is not None
        assert self.old_assigner is not None
        state.update_assignment(self.old_assignment)
        state._slice_assigner = self.old_assigner
        return self.old_assigner

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWDecrementNcMutation']:
        if state.Nc <= 4:
            return None
        new_Nc = state.Nc - 1
        if state.K * new_Nc < len(state.c):
            return None

        evicted_real = [u for u in range(len(state.c)) if state.x[u] == state.Nc - 1]
        
        occupied = set(zip(state.c, state.x))
        empty_slots = []
        for c in range(state.K):
            for x in range(new_Nc):
                if (c, x) not in occupied:
                    empty_slots.append((c, x))

        if len(evicted_real) > len(empty_slots):
            return None
            
        chosen_slots = rng.choice(len(empty_slots), size=len(evicted_real), replace=False)
        evicted_swaps = [(evicted_real[i], empty_slots[idx]) for i, idx in enumerate(chosen_slots)]
        return cls(evicted_swaps)


class IHWSwapCoresMutation(IHWMutation):
    def __init__(self, core_a: int, core_b: int):
        self.core_a = core_a
        self.core_b = core_b

    def do(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner, graph: gt.Graph) -> DeltaOptimalSliceAssigner:
        mask_a = (state.c == self.core_a)
        mask_b = (state.c == self.core_b)
        
        moved_nodes = list(np.where(mask_a | mask_b)[0])
        old_values = {node: (int(state.c[node]), int(state.x[node])) for node in moved_nodes}
        
        state.c[mask_a] = self.core_b
        state.c[mask_b] = self.core_a
        
        slice_assigner.delta_assign_slices(
            state=state,
            moved_nodes=moved_nodes,
            graph=graph,
            old_core_and_local_of_moved_nodes=old_values
        )
        return slice_assigner

    def undo(self, state: SAIHWState, slice_assigner: DeltaOptimalSliceAssigner) -> DeltaOptimalSliceAssigner:
        mask_a = (state.c == self.core_a)
        mask_b = (state.c == self.core_b)
        
        state.c[mask_a] = self.core_b
        state.c[mask_b] = self.core_a
        
        slice_assigner.undo_assign_slices()
        return slice_assigner

    @classmethod
    def create_random(cls, state: SAIHWState, rng: np.random.Generator) -> Optional['IHWSwapCoresMutation']:
        if state.K < 2:
            return None
        core_a, core_b = rng.choice(state.K, size=2, replace=False)
        return cls(int(core_a), int(core_b))
