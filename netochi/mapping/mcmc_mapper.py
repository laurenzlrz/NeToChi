import graph_tool.all as gt
import numpy as np
from netochi.mapping.hardware_config import HardwareConfig
from netochi.mapping.likelihood_state import MappingState
from netochi.pipeline.core import BaseMapper, IFixedHardwareMapper, FixedHardwareInput

class Section5SBM_MCMCMapper(BaseMapper, IFixedHardwareMapper):
    """Mapper using Simulated Annealing to optimize the mapping objective."""
    def map_fixed_hardware(self, mapping_input: FixedHardwareInput, iterations=1000, initial_temp=1.0, seed=None) -> MappingState:
        """Execute Simulated Annealing optimization for fixed hardware."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        state.init_random(seed=seed)
        
        # Determine the objective function to use
        # Fallback to log likelihood if not provided (though FixedHardwareInput should have it)
        objective = mapping_input.objective
        
        def evaluate_energy(st):
            if objective:
                return objective.evaluate(st)
            return -st.log_likelihood()
            
        current_energy = evaluate_energy(state)
        best_energy = current_energy
        
        # Save best state
        best_c = state.c.copy()
        best_x = state.x.copy()
        best_s = state.s.copy()
        
        rng = np.random.default_rng(seed)
        
        for i in range(iterations):
            # Exponential cooling schedule
            T = initial_temp * ((0.01 / initial_temp) ** (i / max(1, iterations - 1)))
            
            # Propose a move
            move_type = rng.integers(0, 2)
            node = rng.integers(0, state.N)
            old_energy = current_energy
            
            if move_type == 0:
                # Swap core and local address with another node
                node2 = rng.integers(0, state.N)
                if node == node2:
                    continue
                    
                old_c1, old_x1 = state.c[node], state.x[node]
                old_c2, old_x2 = state.c[node2], state.x[node2]
                
                state.c[node], state.x[node] = old_c2, old_x2
                state.c[node2], state.x[node2] = old_c1, old_x1
                
                new_energy = evaluate_energy(state)
                
                if new_energy < old_energy or rng.random() < np.exp((old_energy - new_energy) / T):
                    current_energy = new_energy
                else:
                    # Revert
                    state.c[node], state.x[node] = old_c1, old_x1
                    state.c[node2], state.x[node2] = old_c2, old_x2
                    
            else:
                # Mutate slice assignment at a random distance
                d = rng.integers(1, state.config.max_distance + 1)
                n_slices = state.config.num_slices_at_distance(d)
                
                old_s = state.s[node, d]
                new_s = rng.integers(0, n_slices)
                
                if old_s == new_s:
                    continue
                    
                state.s[node, d] = new_s
                new_energy = evaluate_energy(state)
                
                if new_energy < old_energy or rng.random() < np.exp((old_energy - new_energy) / T):
                    current_energy = new_energy
                else:
                    state.s[node, d] = old_s

            if current_energy < best_energy:
                best_energy = current_energy
                best_c = state.c.copy()
                best_x = state.x.copy()
                best_s = state.s.copy()
                
        # Restore best mapping found
        state.c = best_c
        state.x = best_x
        state.s = best_s
        return state
