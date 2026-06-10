from typing import List, Any, Tuple, cast
from netochi.mapping.interfaces import (
    MappingState, 
    BaseMosaicMappingState, 
    MosaicNetworkMappingState, 
    MosaicHWMappingState,
    BaseMapper
)
from netochi.input_generator.interfaces import BaseInputFactory, MosaicHWMappingInput, MappingInput

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiFactory
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory
from netochi.input_generator.swta_factory import SwtaFactory

from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.mcmc.mcmc_mapper import MCMCMapper
from netochi.mapping.mcmc.joint_inference_mapper import JointInferenceMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapper

from netochi.mapping.three_step_mapping.hcd_pca_opt_three_step_mapper import HcdPcaOptThreeStepMapper
from netochi.pipeline.runner import (
    PipelineRunner, 
    ExperimentTask, 
    Evaluator, 
    MosaicGroundTruthBaselineProvider, 
    MapperBaselineProvider,
    BaseBaselineProvider
)
from netochi.pipeline.metrics import ObjectiveMetric
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
    mosaic_factories: List[BaseInputFactory[MosaicHWMappingInput[Any]]] = [
        MosaicNetworkFactory(hw_config=hw_small, probability=p, seed=42) for p in probabilities
    ]
    er_factories: List[BaseInputFactory[MosaicHWMappingInput[Any]]] = [
        ErdosRenyiFactory(hw_config=hw_small, n=60, probability=p, seed=42) for p in probabilities
    ]
    swta_factories: List[BaseInputFactory[MosaicHWMappingInput[Any]]] = [
        SwtaFactory(hw_config=hw_small, num_clusters=4, neurons_per_cluster=15, seed=42)
    ]

    # 2. Define Evaluators
    log_likelihood_obj: LogLikelihoodObjective[MosaicHWMappingInput[Any], Any] = LogLikelihoodObjective()
    inconsistency_obj: InconsistencyObjective[MosaicHWMappingInput[Any], Any] = InconsistencyObjective()
    hw_size_obj: MosaicHardwareSizeObjective[MosaicHWMappingInput[Any]] = MosaicHardwareSizeObjective()


    standard_evaluator: Evaluator[BaseMosaicMappingState[Any], MappingState[Any]] = Evaluator(
        metrics=[
            ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], MappingState[Any]], log_likelihood_obj)),
            ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], MappingState[Any]], inconsistency_obj))
        ]
    )

    hw_evaluator: Evaluator[BaseMosaicMappingState[Any], MappingState[Any]] = Evaluator(
        metrics=[
            ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], MappingState[Any]], log_likelihood_obj)),
            ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], MappingState[Any]], inconsistency_obj)),
            ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], MappingState[Any]], hw_size_obj))
        ]
    )

    # 3. Define Baseline Providers
    gt_baseline = MosaicGroundTruthBaselineProvider()
    random_baseline = MapperBaselineProvider(mapper=RandomMapper(seed=42))

    # 4. Define Tasks
    # We group mappers by their evaluation strategy and inputs
    mappers_std = [
        SimAnnealingMapper()
    ]
    
    # Use a more specific task type internally to avoid Any during creation
    mosaic_tasks: List[ExperimentTask[MosaicHWMappingInput[Any], BaseMosaicMappingState[Any], MappingState[Any]]] = []

    # Add Standard Mappers
    for mapper in mappers_std:
        # Each factory is paired with its appropriate baseline provider
        task_inputs: List[Tuple[BaseInputFactory[MosaicHWMappingInput[Any]], BaseBaselineProvider[MappingState[Any], MosaicHWMappingInput[Any]]]] = []
        for f in mosaic_factories:
            task_inputs.append((f, gt_baseline))
        for f in er_factories:
            task_inputs.append((f, random_baseline))
        for f in swta_factories:
            task_inputs.append((f, random_baseline))

        mosaic_tasks.append(ExperimentTask(
            mapper=cast(BaseMapper[BaseMosaicMappingState[Any], MosaicHWMappingInput[Any]], mapper),
            evaluator=standard_evaluator, 
            inputs=task_inputs
        ))

    # Add Joint Inference Mapper


    # 5. Run Pipeline
    print("=" * 100)
    print("Neuromorphic Mapping Pipeline Execution (Task-Centric)")
    print("=" * 100)

    # Final cast to allow the runner to accept the mosaic-specific tasks
    general_tasks = cast(List[ExperimentTask[MappingInput[Any], MappingState[Any], MappingState[Any]]], mosaic_tasks)
    runner = PipelineRunner(tasks=general_tasks, verbose=True)
    summary = runner.run()

    # 6. Summary Report
    from netochi.pipeline.constants import (
        REPORT_DIVIDER, REPORT_SUBDIVIDER, REPORT_HEADER_BASELINE, REPORT_HEADER_PURE,
        TABLE_HEADER_REL_FORMAT, TABLE_HEADER_RAW_FORMAT, TABLE_ROW_REL_FORMAT, TABLE_ROW_RAW_FORMAT,
        KEY_GRAPH_TYPE, KEY_UNKNOWN, DEFAULT_METRIC_VALUE
    )
    from netochi.objectives.constants import OBJ_NAME_LL, OBJ_NAME_INCONSISTENCY, OBJ_NAME_HW_SIZE

    print("\n" + REPORT_DIVIDER)
    print(REPORT_HEADER_BASELINE)
    print(REPORT_DIVIDER)
    print(TABLE_HEADER_REL_FORMAT)
    print(REPORT_SUBDIVIDER)
    
    for res in summary.results:
        rel_ll = res.metrics.get(OBJ_NAME_LL, DEFAULT_METRIC_VALUE)
        rel_inc = res.metrics.get(OBJ_NAME_INCONSISTENCY, DEFAULT_METRIC_VALUE)
        rel_hw = res.metrics.get(OBJ_NAME_HW_SIZE, DEFAULT_METRIC_VALUE)
        graph_type = res.input_metadata.get(KEY_GRAPH_TYPE, KEY_UNKNOWN)
        print(TABLE_ROW_REL_FORMAT.format(
            mapper=res.mapper_name,
            graph_type=graph_type,
            rel_ll=rel_ll,
            rel_inc=rel_inc,
            rel_hw=rel_hw,
            elapsed=res.execution_time_s
        ))

    print("\n" + REPORT_DIVIDER)
    print(REPORT_HEADER_PURE)
    print(REPORT_DIVIDER)
    print(TABLE_HEADER_RAW_FORMAT)
    print(REPORT_SUBDIVIDER)

    for res in summary.results:
        raw_ll = res.raw_metrics.get(OBJ_NAME_LL, DEFAULT_METRIC_VALUE)
        raw_inc = res.raw_metrics.get(OBJ_NAME_INCONSISTENCY, DEFAULT_METRIC_VALUE)
        raw_hw = res.raw_metrics.get(OBJ_NAME_HW_SIZE, DEFAULT_METRIC_VALUE)
        graph_type = res.input_metadata.get(KEY_GRAPH_TYPE, KEY_UNKNOWN)
        print(TABLE_ROW_RAW_FORMAT.format(
            mapper=res.mapper_name,
            graph_type=graph_type,
            raw_ll=raw_ll,
            raw_inc=raw_inc,
            raw_hw=raw_hw,
            elapsed=res.execution_time_s
        ))
    
    print(REPORT_DIVIDER)
    print(f"Total Experiment Time: {summary.total_time_s:.2f}s")
    print("Experiment Complete.")


if __name__ == '__main__':
    run_experiment()
