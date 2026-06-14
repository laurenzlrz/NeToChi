from typing import List, Any, Tuple, cast
from netochi.mapping.interfaces import (
    MappingState, 
    BaseMosaicMappingState, 
)
from netochi.input_generator.interfaces import BaseInputFactory, MosaicMappingInput, MappingInput
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiFactory
from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper

from netochi.pipeline.runner import (
    PipelineRunner,
    ExperimentTaskOneComb,
    MapperBaselineProvider,
)
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.metrics import ObjectiveMetric
from netochi.objectives.log_likelihood import LogLikelihoodObjective
from netochi.objectives.interfaces import MappingObjective

def run_test() -> None:
    """
    Lightweight test script to verify storage and plotting.
    Uses only two mappers and one graph type to ensure fast execution.
    """
    # --- Hardware Configuration ---
    hw_small = MosaicHardwareConfig(
        nodes_per_router=2,
        neurons_per_core=15,
        router_levels=2,
        slice_factor=2
    )

    # 1. Define the inputs (Small Erdos-Renyi)
    er_factories: List[BaseInputFactory[MosaicMappingInput[Any]]] = [
        ErdosRenyiFactory(hw_config=hw_small, n=30, probability=0.3, seed=42)
    ]

    # 2. Define Evaluators
    log_likelihood_obj: LogLikelihoodObjective[MosaicMappingInput[Any], Any] = LogLikelihoodObjective()
    standard_evaluator: EvaluatorBundle[BaseMosaicMappingState[Any], MappingState[Any]] = EvaluatorBundle(
        metrics=[
            ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], MappingState[Any]], log_likelihood_obj)),
        ]
    )

    # 3. Define Baseline Providers
    random_baseline = MapperBaselineProvider(mapper=RandomMapper(seed=42))

    # 4. Define Tasks (Random and Greedy)
    mappers = [RandomMapper(), GreedyMapper()]
    mosaic_tasks: List[ExperimentTaskOneComb[MosaicMappingInput[Any], BaseMosaicMappingState[Any], MappingState[Any]]] = []

    for mapper in mappers:
        task_inputs: List[Tuple[BaseInputFactory[MosaicMappingInput[Any]], Any]] = []
        for f in er_factories:
            task_inputs.append((f, random_baseline))

        mosaic_tasks.append(ExperimentTaskOneComb(
            mapper=cast(Any, mapper), 
            evaluator=standard_evaluator, 
            inputs=task_inputs
        ))

    # 5. Output Configuration
    output_config = PipelineOutputConfig(
        base_dir="test_results",
        plot_format="png",
        palette=["#0077BB", "#EE7733"] # Custom test palette
    )

    # 6. Run Pipeline
    print("=" * 60)
    print("Running Pipeline Plotting Verification Test")
    print("=" * 60)
    
    general_tasks = cast(List[ExperimentTaskOneComb[MappingInput[Any], MappingState[Any], MappingState[Any]]], mosaic_tasks)
    runner = PipelineRunner(tasks=general_tasks, output_config=output_config, verbose=True)
    summary = runner.run()
    
    print("=" * 60)
    print(f"Test Complete. Total Time: {summary.total_time_s:.2f}s")
    print("=" * 60)

if __name__ == '__main__':
    run_test()
