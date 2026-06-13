from typing import List, Any, Tuple, cast

from netochi.mapping.ilp_mapper import ILPMapper
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
    BaseMapper
)
from netochi.input_generator.interfaces import MosaicMappingInput, MappingInput, HWBaseInputFactory

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiFactory
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory
from netochi.input_generator.swta_factory import SwtaFactory, SwtaGeneratorConfig

from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.greedy_mapper import GreedyMapper
from netochi.mapping.mcmc.mcmc_mapper import MCMCMapper
from netochi.mapping.mcmc.joint_inference_mapper import JointInferenceMapper
from netochi.mapping.three_step_mapping.hcd_pca_opt_three_step_mapper import HcdPcaOptThreeStepMapper
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapper
from netochi.mapping.three_step_mapping.qap_pca_opt_three_step_mapper import QAPPcaOptMapper

from netochi.objectives.obj_unused_connections import UnusedConnectionsObjective
from netochi.pipeline.runner.runner import (
    PipelineRunner,
    ExperimentTaskBase,
    ExperimentTaskRun,
)
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle
from netochi.pipeline.metrics import ObjectiveMetric
from netochi.objectives.obj_log_likelihood import LogLikelihoodObjective
from netochi.objectives.obj_inconsistency import InconsistencyObjective, InconsistencyRelativeObjective
from netochi.objectives.obj_hardware_size import MosaicHardwareSizeObjective
from netochi.objectives.interfaces import MappingObjective
from netochi.definitions.constants import KEY_GRAPH_TYPE, KEY_UNKNOWN, DEFAULT_METRIC_VALUE, REPORT_DIVIDER, \
    REPORT_SUBDIVIDER, REPORT_HEADER_BASELINE, REPORT_HEADER_PURE, TABLE_ROW_REL_FORMAT, TABLE_ROW_RAW_FORMAT, \
    TABLE_HEADER_REL_FORMAT, TABLE_HEADER_RAW_FORMAT, OBJ_NAME_LL, OBJ_NAME_INCONSISTENCY, OBJ_NAME_HW_SIZE
from runner.baseline_provider import RandomMosaicBaselineProvider, MosaicGroundTruthBaselineProvider

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
    SimAnnealingMapper(),
    QAPPcaOptMapper(),
    GreedyMapper(),
    ILPMapper(),
]

SEED = 42

# ==============================================================================

def define_task_inputs() -> List[PipelineRunner]:

    # 1. Define the inputs (Factories)
    probabilities = [0.1, 0.5]
    mosaic_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        MosaicNetworkFactory(hw_config=HW_SMALL, probability=p, seed=SEED) for p in probabilities
    ]
    er_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        ErdosRenyiFactory(hw_config=HW_SMALL, n=HW_SMALL.total_neurons, probability=p, seed=SEED) for p in probabilities
    ]
    swta_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        SwtaFactory(hw_config=HW_SMALL, config=SwtaGeneratorConfig(num_clusters=10,
                                                                   neurons_per_cluster=10,
                                                                   inhibitory_ratio=0.2,
                                                                   p_neighbor=0.1,
                                                                   p_e_to_i=0.2,
                                                                   p_i_to_e=0.8, seed=None, ))
    ]

    # 2. Define Baseline Providers
    random_baseline = RandomMosaicBaselineProvider()
    gt_baseline = MosaicGroundTruthBaselineProvider()
    metrics = [ObjectiveMetric(objective=cast(MappingObjective[BaseMosaicMappingState[Any], BaseMosaicMappingState[Any]], objective)) for objective in OBJECTIVES]

    mosaic_evaluator_baseline_bundles = EvaluatorBundle(metrics_w_baselines=[(metric, gt_baseline) for metric in metrics])
    other_gen_evaluator_baseline_bundles = EvaluatorBundle(metrics_w_baselines=[(metric, random_baseline) for metric in metrics])

    mosaic_task_runs = [ExperimentTaskRun(evaluator_bundle=mosaic_evaluator_baseline_bundles,
                                          mapper=mapper) for mapper in MAPPERS]

    other_task_runs = [ExperimentTaskRun(evaluator_bundle=other_gen_evaluator_baseline_bundles,
                                         mapper=mapper) for mapper in MAPPERS]

    mosaic_tasks = ExperimentTaskBase(input_generators=mosaic_factories,
                                       evaluator_mapper_bundles=mosaic_task_runs)

    er_tasks = ExperimentTaskBase(input_generators=er_factories,
                                  evaluator_mapper_bundles=other_task_runs)

    swta_tasks = ExperimentTaskBase(input_generators=swta_factories,
                                    evaluator_mapper_bundles=other_task_runs)

    runner = [PipelineRunner(tasks=[task]) for task in [mosaic_tasks, er_tasks, swta_tasks]]
    return runner


def run_experiment() -> None:

    # === 1. define mosaic tasks ===
    runners = define_task_inputs()

    # === 2. Run Pipeline ===
    print("=" * 100)
    print("Neuromorphic Mapping Pipeline Execution")
    print("=" * 100)

    for run in runners:
        summary = run.run()

        # === 3. Summary Report ===
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
