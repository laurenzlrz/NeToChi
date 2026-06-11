from typing import List, Any, Tuple, cast
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
    BaseMapper
)
from netochi.input_generator.interfaces import MosaicHWMappingInput, MappingInput, HWBaseInputFactory

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiFactory
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory
from netochi.input_generator.swta_factory import SwtaFactory

from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.mcmc.mcmc_mapper import MCMCMapper
from netochi.mapping.mcmc.joint_inference_mapper import JointInferenceMapper
from netochi.mapping.qap_mapper import QAPMapper
from netochi.mapping.three_step_mapping.hcd_pca_opt_three_step_mapper import HcdPcaOptThreeStepMapper
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapper

from netochi.objectives.obj_inconsistency_relative import InconsistencyRelativeObjective
from netochi.objectives.obj_unused_connections import UnusedConnectionsObjective
from netochi.pipeline.runner import (
    PipelineRunner, 
    ExperimentTask, 
    Evaluator, 
    MosaicGroundTruthBaselineProvider, 
    MapperBaselineProvider,
    BaseBaselineProvider
)
from netochi.pipeline.metrics import ObjectiveMetric
from netochi.objectives.obj_log_likelihood import LogLikelihoodObjective
from netochi.objectives.obj_inconsistency import InconsistencyObjective
from netochi.objectives.obj_hardware_size import MosaicHardwareSizeObjective
from netochi.objectives.interfaces import MappingObjective
from netochi.pipeline.constants import (
    REPORT_DIVIDER, REPORT_HEADER_BASELINE, REPORT_HEADER_PURE,
    KEY_GRAPH_TYPE, KEY_UNKNOWN, DEFAULT_METRIC_VALUE
)



# ======================= CONFIGURE PIPELINE HERE =============================

HW_SMALL = MosaicHardwareConfig(
    nodes_per_router=2,
    neurons_per_core=15,
    router_levels=2,
    slice_factor=2
)

OBJECTIVES = [
    InconsistencyObjective(),
    InconsistencyRelativeObjective(),
    MosaicHardwareSizeObjective(),
    UnusedConnectionsObjective(),
    LogLikelihoodObjective()
]

HW_CONFIGS = [
    HW_SMALL
]

MAPPERS = [
    SimAnnealingMapper()
]

SEED = 42

# ==============================================================================

def define_task_inputs() -> List[Tuple[HWBaseInputFactory, BaseBaselineProvider]]:

    # 1. Define the inputs (Factories)
    probabilities = [0.1, 0.5]
    mosaic_factories: List[HWBaseInputFactory[MosaicHWMappingInput[Any]]] = [
        MosaicNetworkFactory(hw_config=HW_SMALL, probability=p, seed=SEED) for p in probabilities
    ]
    er_factories: List[HWBaseInputFactory[MosaicHWMappingInput[Any]]] = [
        ErdosRenyiFactory(hw_config=HW_SMALL, n=HW_SMALL.total_neurons, probability=p, seed=SEED) for p in probabilities
    ]
    swta_factories: List[HWBaseInputFactory[MosaicHWMappingInput[Any]]] = [
        SwtaFactory(hw_config=HW_SMALL, seed=SEED)
    ]

    # 2. Define Baseline Providers
    gt_baseline = MosaicGroundTruthBaselineProvider()
    random_baseline = MapperBaselineProvider(mapper=RandomMapper(seed=SEED))

    # 3. generate task inputs
    task_inputs: List[Tuple[HWBaseInputFactory[MosaicHWMappingInput[Any]], BaseBaselineProvider[
        BaseMosaicMappingState[Any], MosaicHWMappingInput[Any]]]] = []
    for f in mosaic_factories:
        task_inputs.append((f, gt_baseline))
    for f in er_factories:
        task_inputs.append((f, random_baseline))
    for f in swta_factories:
        task_inputs.append((f, random_baseline))
    return task_inputs


def define_evaluators() -> Evaluator[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]:
    metrics = [ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]], objective)) for objective in OBJECTIVES]
    standard_evaluator: Evaluator[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]] = Evaluator(metrics=metrics)
    return standard_evaluator


def print_metrics(summary, metric_keys, raw: bool):

    col_width = max([len(key_name) for key_name in metric_keys])
    header_metrics_str = " | ".join([f"{key:<{col_width}}" for key in metric_keys])
    header = f"{'Mapper':<{col_width}} | {'Graph Type':<{col_width}} | {header_metrics_str} | {'Elapsed (s)':<{col_width}}"

    report_divider = "=" * (col_width * (len(metric_keys) + 3))
    report_subdivider = "-" * (col_width * (len(metric_keys) + 3))

    print("\n" + report_divider)
    if raw:
        print(REPORT_HEADER_PURE)
    else:
        print(REPORT_HEADER_BASELINE)
    print(report_divider)
    print(header)
    print(report_subdivider)
    for res in summary.results:
        graph_type = res.input_metadata.get(KEY_GRAPH_TYPE, KEY_UNKNOWN)
        metrics_str_list = []
        for key in metric_keys:
            if raw:
                val = res.raw_metrics.get(key, DEFAULT_METRIC_VALUE)
            else:
                val = res.metrics.get(key, DEFAULT_METRIC_VALUE)
            if isinstance(val, float):
                metrics_str_list.append(f"{val:<{col_width}.4f}")
            else:
                metrics_str_list.append(f"{str(val):<{col_width}}")
        metrics_row_str = " | ".join(metrics_str_list)
        print(f"{res.mapper_name:<{col_width}} | {graph_type:<{col_width}} | {metrics_row_str} | {res.execution_time_s:<12.2f}")

def run_experiment() -> None:

    # === 1. define mosaic tasks ===
    standard_evaluator = define_evaluators()
    task_inputs = define_task_inputs()

    mosaic_tasks: List[ExperimentTask[MosaicHWMappingInput[Any], BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]] = []
    for mapper in MAPPERS:
        mosaic_tasks.append(ExperimentTask(
            mapper=cast(BaseMapper[BaseMosaicMappingState[Any], MosaicHWMappingInput[Any]], mapper),
            evaluator=standard_evaluator, 
            inputs=task_inputs
        ))

    # === 2. Run Pipeline ===
    print("=" * 100)
    print("Neuromorphic Mapping Pipeline Execution")
    print("=" * 100)

    # Final cast to allow the runner to accept the mosaic-specific tasks
    general_tasks = cast(List[ExperimentTask[MappingInput[Any], BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]]], mosaic_tasks)
    runner = PipelineRunner(tasks=general_tasks, verbose=True)
    summary = runner.run()

    # === 3. Summary Report ===
    # Dynamically extract all unique metric names from the results
    metric_keys_dict = {}
    for res in summary.results:
        for key in res.metrics.keys():
            metric_keys_dict[key] = None
    metric_keys = list(metric_keys_dict.keys())

    # Build dynamic format strings for the headers

    # Print Relative Metrics
    print_metrics(summary=summary, metric_keys=metric_keys, raw=False)

    # Print Raw Metrics
    print_metrics(summary=summary, metric_keys=metric_keys, raw=True)

    print(REPORT_DIVIDER)
    print(f"Total Experiment Time: {summary.total_time_s:.2f}s")
    print("Experiment Complete.")


if __name__ == '__main__':
    run_experiment()
