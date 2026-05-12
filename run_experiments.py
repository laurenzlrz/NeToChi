from typing import List, Any, Tuple, cast
from netochi.mapping.interfaces import (
    MappingState, 
    BaseMosaicMappingState, 
    MosaicNetworkMappingState, 
    MosaicHWMappingState,
    BaseMapper
)
from netochi.input_generator.interfaces import BaseInputFactory, MosaicMappingInput, MappingInput

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiFactory
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory
from netochi.input_generator.swta_factory import SwtaFactory

from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.mcmc.mcmc_mapper import MCMCMapper
from netochi.mapping.mcmc.joint_inference_mapper import JointInferenceMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.mapping.hierarchical_community_detection.hybrid_mapper import HybridMapper

from netochi.pipeline.runner import (
    PipelineRunner, 
    ExperimentTask, 
    Evaluator, 
    MosaicGroundTruthBaselineProvider, 
    MapperBaselineProvider,
    BaseBaselineProvider
)
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.metrics import ObjectiveMetric, InconsistencyPercentageMetric
from netochi.objectives.log_likelihood import LogLikelihoodObjective
from netochi.objectives.inconsistency import InconsistencyObjective
from netochi.objectives.hardware_size import MosaicHardwareSizeObjective
from netochi.objectives.interfaces import MappingObjective


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
    mosaic_factories: List[BaseInputFactory[MosaicMappingInput[Any]]] = [
        MosaicNetworkFactory(hw_config=hw_small, probability=p, seed=42) for p in probabilities
    ]
    er_factories: List[BaseInputFactory[MosaicMappingInput[Any]]] = [
        ErdosRenyiFactory(hw_config=hw_small, n=60, probability=p, seed=42) for p in probabilities
    ]
    swta_factories: List[BaseInputFactory[MosaicMappingInput[Any]]] = [
        SwtaFactory(hw_config=hw_small, num_clusters=4, neurons_per_cluster=15, seed=42)
    ]

    # 2. Define Evaluators
    log_likelihood_obj: LogLikelihoodObjective[MosaicMappingInput[Any], Any] = LogLikelihoodObjective()
    inconsistency_obj: InconsistencyObjective[MosaicMappingInput[Any], Any] = InconsistencyObjective()
    hw_size_obj: MosaicHardwareSizeObjective[MosaicMappingInput[Any]] = MosaicHardwareSizeObjective()


    standard_evaluator: Evaluator[BaseMosaicMappingState[Any], MosaicNetworkMappingState[Any]] = Evaluator(
        metrics=[
            ObjectiveMetric(objective=log_likelihood_obj),
            ObjectiveMetric(objective=inconsistency_obj),
            InconsistencyPercentageMetric(objective=inconsistency_obj)
        ]
    )

    hw_evaluator: Evaluator[BaseMosaicMappingState[Any], BaseMosaicMappingState[MosaicMappingInput[Any]]] = Evaluator(
        metrics=[
            ObjectiveMetric(objective=log_likelihood_obj),
            ObjectiveMetric(objective=inconsistency_obj),
            InconsistencyPercentageMetric(objective=inconsistency_obj),
            ObjectiveMetric(objective=hw_size_obj)
        ]
    )

    # 3. Define Baseline Providers
    gt_baseline = MosaicGroundTruthBaselineProvider()
    random_baseline = MapperBaselineProvider(mapper=RandomMapper(seed=42))

    # 4. Define Tasks
    # We group mappers by their evaluation strategy and inputs
    mappers_std = [
        #HybridMapper(),
        RandomMapper(),
        GreedyMapper(),
        #MCMCMapper(objective=log_likelihood_obj, iterations=200, verbose=False),
        #QAPMapper(),
        
    ]
    
    # Use a more specific task type internally to avoid Any during creation
    mosaic_tasks: List[ExperimentTask[MosaicMappingInput[Any], BaseMosaicMappingState[Any], MappingState[Any]]] = []

    # Add Standard Mappers
    for mapper in mappers_std:
        # Each factory is paired with its appropriate baseline provider
        task_inputs: List[Tuple[BaseInputFactory[MosaicMappingInput[Any]], BaseBaselineProvider[MappingState[Any], MosaicMappingInput[Any]]]] = []
        for f in mosaic_factories:
            task_inputs.append((f, gt_baseline))
        for f in er_factories:
            task_inputs.append((f, random_baseline))
        for f in swta_factories:
            task_inputs.append((f, random_baseline))

        mosaic_tasks.append(ExperimentTask(
            mapper=mapper, 
            evaluator=standard_evaluator, 
            inputs=task_inputs
        ))

    # Add Joint Inference Mapper
    inference_inputs: List[Tuple[BaseInputFactory[MosaicMappingInput[Any]], BaseBaselineProvider[MappingState[Any], MosaicMappingInput[Any]]]] = []
    for f in mosaic_factories:
        inference_inputs.append((f, gt_baseline))
    for f in er_factories:
        inference_inputs.append((f, random_baseline))
    for f in swta_factories:
        inference_inputs.append((f, random_baseline))

    #mosaic_tasks.append(ExperimentTask(
    #    mapper=JointInferenceMapper(objective=log_likelihood_obj, iterations=200, verbose=False),
    #    evaluator=hw_evaluator,
    #    inputs=inference_inputs
    #))

    # 5. Run Pipeline
    print("=" * 100)
    print("Neuromorphic Mapping Pipeline Execution (Task-Centric)")
    print("=" * 100)

    output_config = PipelineOutputConfig(
        base_dir="results",
        plot_format="png"
    )

    # Pass baseline providers for explicit benchmarking
    baseline_benchmarks = {}
    for f in mosaic_factories:
        baseline_benchmarks[f] = gt_baseline
    for f in er_factories:
        baseline_benchmarks[f] = random_baseline
    for f in swta_factories:
        baseline_benchmarks[f] = random_baseline

    # Final cast to allow the runner to accept the mosaic-specific tasks
    general_tasks = cast(List[ExperimentTask[MappingInput[Any], MappingState[Any], MappingState[Any]]], mosaic_tasks)
    runner = PipelineRunner(
        tasks=general_tasks, 
        baselines=baseline_benchmarks,
        output_config=output_config, 
        verbose=True
    )
    summary = runner.run()

    # 6. Summary Report
    from netochi.pipeline.reporter import SummaryReporter
    SummaryReporter.print_report(summary)


if __name__ == '__main__':
    run_experiment()
