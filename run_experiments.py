from typing import List, Any

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiFactory
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory
from netochi.input_generator.wta_factory import WTAFactory

from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.hybrid_mapper import HybridMapper
from netochi.mapping.mcmc.mcmc_mapper import MCMCMapper
from netochi.mapping.mcmc.joint_inference_mapper import JointInferenceMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.mapping.ilp_mapper import ILPMapper

from netochi.pipeline.runner import PipelineRunner
from netochi.pipeline.metrics import ObjectiveMetric
from netochi.objectives.log_likelihood import LogLikelihoodObjective
from netochi.objectives.inconsistency import InconsistencyObjective


def run_experiment():
    # --- Hardware Configuration: Small (4 cores × 15 = 60 neurons) ---
    hw_small = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=15,
        router_levels=2,
        slice_factor=2
    )

    # --- Hardware Configuration: Large (8 cores × 75 = 600 neurons) ---
    hw_large = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=75,
        router_levels=3,
        slice_factor=2
    )

    # 1. Define the inputs (Factories)
    probabilities = [0.1, 0.3, 0.5]
    factories: List[Any] = []

    # Add SBM-based Mosaic factories
    for p in probabilities:
        factories.append(MosaicNetworkFactory(hw_config=hw_small, probability=p, seed=42))
        factories.append(MosaicNetworkFactory(hw_config=hw_large, probability=p, seed=42))

    # Add Erdős-Rényi factories
    for p in probabilities:
        factories.append(ErdosRenyiFactory(hw_config=hw_small, n=60, probability=p, seed=42))

    # Add WTA factories
    for p in probabilities:
        factories.append(WTAFactory(hw_config=hw_small, probability=p, seed=42))

    # 2. Define the mappers to evaluate (Instantiated)
    log_likelihood_obj = LogLikelihoodObjective()
    
    mappers: List[Any] = [
        RandomMapper(),
        GreedyMapper(),
        HybridMapper(),
        QAPMapper(),
        MCMCMapper(objective=log_likelihood_obj, iterations=1000, verbose=False),
        # JointInferenceMapper(objective=log_likelihood_obj, hw_template=hw_small, verbose=False),
    ]

    # 3. Define the metrics (Instantiated using ObjectiveMetric adapter)
    metrics = [
        ObjectiveMetric(objective=log_likelihood_obj),
        ObjectiveMetric(objective=InconsistencyObjective())
    ]

    # 4. Run Pipeline
    print("=" * 80)
    print("Neuromorphic Mapping Pipeline Execution (Modern Architecture)")
    print("=" * 80)

    runner = PipelineRunner(
        factories=factories,
        mappers=mappers,
        metrics=metrics,
        verbose=True
    )

    summary = runner.run()

    # 5. Summary Report
    print("\n" + "=" * 80)
    print(f"{'Mapper':<25} | {'Graph':<20} | {'Log-Likelihood':<15} | {'Time (s)':<10}")
    print("-" * 80)
    
    for res in summary.results:
        ll = res.metrics.get("LogLikelihoodObjective", 0.0)
        inconsistency = res.metrics.get("InconsistencyObjective", 0.0)
        graph_type = res.input_metadata.get("graph_type", "Unknown")
        print(f"{res.mapper_name:<25} | {graph_type:<20} | {ll:<15.2f} | {res.execution_time_s:<10.3f}")
    
    print("=" * 80)
    print(f"Total Experiment Time: {summary.total_time_s:.2f}s")
    print("Experiment Complete.")


if __name__ == '__main__':
    run_experiment()
