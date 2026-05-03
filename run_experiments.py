from netochi.mapping.hardware_config import HardwareConfig
from netochi.pipeline.core import BaseMapper, IFixedHardwareMapper, FixedHardwareInput
from netochi.pipeline.runner import PipelineRunner
from netochi.pipeline.metrics import LogLikelihoodMetric, InconsistenciesMetric, ConsistencyPercentageMetric
from netochi.network_generator.factories import RandomNetworkFactory

from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.hybrid_mapper import HybridMapper
from netochi.mapping.mcmc_mapper import Section5SBM_MCMCMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.mapping.ilp_mapper import ILPMapper
from netochi.mapping.likelihood_state import MappingState

class GroundTruthMapper(BaseMapper, IFixedHardwareMapper):
    """Special mapper that evaluates ground truth assignments from metadata."""
    def map_fixed_hardware(self, mapping_input: FixedHardwareInput) -> MappingState:
        """Extract and return the ground truth mapping state."""
        state = MappingState(mapping_input.graph, mapping_input.hw_config)
        ground_truth = mapping_input.metadata.get("ground_truth_assignments")
        if ground_truth is not None:
            for i in range(state.N):
                state.c[i] = i // mapping_input.hw_config.neurons_per_core
                state.x[i] = i % mapping_input.hw_config.neurons_per_core
                for d in range(1, mapping_input.hw_config.max_distance + 1):
                    state.s[i, d] = ground_truth[i, d]
        return state

def run_experiment():
    # --- Hardware Configuration: 60 neurons (4 cores × 15) ---
    hw_small = HardwareConfig(
        nodes_per_router=2,
        neurons_per_core=15,
        router_levels=2,   # 4 cores, 60 neurons total
        slice_factor=2
    )

    # --- Hardware Configuration: 600 neurons (8 cores × 75) ---
    hw_large = HardwareConfig(
        nodes_per_router=2,
        neurons_per_core=75,
        router_levels=3,   # 8 cores, 600 neurons total
        slice_factor=2
    )

    # 1. Define the inputs — one factory per (hw_config, probability) pair
    probabilities = [0.1, 0.3, 0.5, 0.8, 1.0]
    factories = [
        RandomNetworkFactory(hw_config=hw, edge_probability=p)
        for hw in [hw_small, hw_large]
        for p in probabilities
    ]

    # 2. Define the mappers to evaluate
    mappers = [
        GroundTruthMapper,
        RandomMapper,
        GreedyMapper,
        HybridMapper,
        QAPMapper,
        ILPMapper,
        Section5SBM_MCMCMapper
    ]

    # 3. Define the metrics to record
    metrics = [
        LogLikelihoodMetric,
        InconsistenciesMetric,
        ConsistencyPercentageMetric
    ]

    # 4. Run Pipeline
    pipeline = PipelineRunner(factories=factories, mapper_classes=mappers, metric_classes=metrics)
    print("=" * 80)
    print("SBM Mapping Pipeline Execution")
    print("=" * 80)
    pipeline.run()
    print("\nExperiment Complete.")

if __name__ == '__main__':
    run_experiment()
