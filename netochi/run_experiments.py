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
from netochi.pipeline.archiver import SummaryArchiver
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.pipeline_consumer import PipelineConsumer
from netochi.pipeline.plotter import PipelinePlotter
from netochi.pipeline.reporter import SummaryReporter
from netochi.pipeline.runner.runner import (
    PipelineRunner,
    ExperimentTaskBase,
    ExperimentTaskRun, TaskBundle,
)
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle
from netochi.pipeline.metrics import ObjectiveMetric
from netochi.objectives.obj_log_likelihood import LogLikelihoodObjective
from netochi.objectives.obj_inconsistency import InconsistencyObjective, InconsistencyRelativeObjective
from netochi.objectives.obj_hardware_size import MosaicHardwareSizeObjective
from netochi.objectives.interfaces import MappingObjective
from netochi.pipeline.runner.baseline_provider import RandomMosaicBaselineProvider, MosaicGroundTruthBaselineProvider

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
    #SimAnnealingMapper(),
    QAPPcaOptMapper(),
    RandomMapper(),
    GreedyMapper(),
    #ILPMapper(),
]

SEED = 42

PIPELINE_CONFIG = PipelineOutputConfig()

CONSUMERS: List[PipelineConsumer] = [
    SummaryReporter(config=PIPELINE_CONFIG),
    PipelinePlotter(config=PIPELINE_CONFIG),
    SummaryArchiver(config=PIPELINE_CONFIG),
]

# ==============================================================================

def define_task_inputs() -> PipelineRunner:

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
    tasks = [mosaic_tasks,
             #er_tasks,
             #swta_tasks
             ]

    bundles = [TaskBundle(tasks=[task], consumer=CONSUMERS) for task in tasks]

    runner = PipelineRunner(bundles=bundles)
    return runner


def run_experiment() -> None:

    # === 1. define mosaic tasks ===
    runner = define_task_inputs()

    # === 2. Run Pipeline ===
    print("=" * 100)
    print("Neuromorphic Mapping Pipeline Execution")
    print("=" * 100)

    summaries = runner.run()


if __name__ == '__main__':
    run_experiment()
