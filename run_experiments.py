from typing import List, Any, Sequence
from netochi.mapping.interfaces import BaseMapper, MosaicNetworkMappingState
from netochi.input_generator.interfaces import BaseInputFactory, MosaicMappingInput
from netochi.pipeline.interfaces import MappingMetric, MappingMetric

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiFactory
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory
from netochi.input_generator.wta_factory import WTAFactory

from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.hybrid_mapper import HybridMapper
from netochi.mapping.mcmc.mcmc_mapper import MCMCMapper
from netochi.mapping.mcmc.joint_inference_mapper import JointInferenceMapper
from netochi.mapping.ilp_mapper import ILPMapper
from netochi.mapping.qap_mapper import QAPMapper

from netochi.pipeline.runner import PipelineRunner
from netochi.pipeline.metrics import ObjectiveMetric
from netochi.objectives.log_likelihood import LogLikelihoodObjective
from netochi.objectives.inconsistency import InconsistencyObjective


def run_experiment() -> None:
    # --- Hardware Configuration: Small (4 cores × 15 = 60 neurons) ---
    hw_small = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=15,
        router_levels=2,
        slice_factor=2
    )

    # 1. Define the inputs (Factories)
    probabilities = [0.1, 0.5]
    factories: List[BaseInputFactory[MosaicMappingInput[Any]]] = []

    # Add SBM-based Mosaic factories (These provide a pre_assignment/baseline)
    for p in probabilities:
        factories.append(MosaicNetworkFactory(hw_config=hw_small, probability=p, seed=42))

    # Add Erdős-Rényi factories (No baseline provided)
    for p in probabilities:
        factories.append(ErdosRenyiFactory(hw_config=hw_small, n=60, probability=p, seed=42))

    # 2. Define the mappers to evaluate
    log_likelihood_obj: LogLikelihoodObjective[MosaicMappingInput[Any], Any] = LogLikelihoodObjective()
    
    mappers: List[BaseMapper[Any, Any]] = [
        RandomMapper(),
        GreedyMapper(),
        MCMCMapper(objective=log_likelihood_obj, iterations=200, verbose=False),
        QAPMapper(),
        HybridMapper(),
        JointInferenceMapper(objective=log_likelihood_obj, iterations=200, verbose=False),
    ]

    # 3. Define the metrics (Now baseline-aware)
    metrics: List[MappingMetric[Any, Any]] = [
        ObjectiveMetric(objective=log_likelihood_obj),
        ObjectiveMetric(objective=InconsistencyObjective())
    ]

    # 4. Run Pipeline
    print("=" * 80)
    print("Neuromorphic Mapping Pipeline Execution (Baseline-Aware)")
    print("=" * 80)

    runner = PipelineRunner[MosaicMappingInput[Any], MosaicNetworkMappingState[Any]](
        factories=factories,
        mappers=mappers,
        metrics=metrics,
        verbose=True
    )

    summary = runner.run()

    # 5. Summary Report
    print("\n" + "=" * 80)
    print(f"{'Mapper':<25} | {'Graph':<20} | {'Rel-Likelihood':<15} | {'Rel-Incons':<10} | {'Time (s)':<10}")
    print("-" * 80)
    
    for res in summary.results:
        # Note: These values are now relative to the baseline if it existed
        rel_ll = res.metrics.get("LogLikelihoodObjective", 0.0)
        rel_inc = res.metrics.get("InconsistencyObjective", 0.0)
        graph_type = res.input_metadata.get("graph_type", "Unknown")
        
        print(f"{res.mapper_name:<25} | {graph_type:<20} | {rel_ll:<15.2f} | {rel_inc:<10.0f} | {res.execution_time_s:<10.3f}")
    
    print("=" * 80)
    print(f"Total Experiment Time: {summary.total_time_s:.2f}s")
    print("Experiment Complete.")


if __name__ == '__main__':
    run_experiment()
