from typing import List, Any, Tuple, cast, Optional

from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
    BaseMapper, MappingState
)
from netochi.input_generator.interfaces import MosaicMappingInput, MappingInput, HWBaseInputFactory

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
from netochi.input_generator.swta_factory import SwtaGeneratorConfig

from netochi.mapping.random_mapper import RandomMapperConfig
from netochi.mapping.greedy_mapper import GreedyMapperConfig
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapperConfig
from netochi.mapping.three_step_mapping.qap_pca_opt_three_step_mapper import QAPPcaOptMapperConfig
from netochi.mapping.ilp_mapper import ILPMapperConfig
from netochi.mapping.sa_ihw_config import SimAnnealingIHWConfig

from netochi.objectives.obj_unused_connections import UnusedConnectionsObjectiveConfig
from netochi.pipeline import BasePipelineRunner
from netochi.pipeline.archiver import SummaryArchiverConfig
from netochi.pipeline.config import PipelineOutputConfig, PipelineOutput
from netochi.pipeline.interfaces import PipelineConsumer, MappingStateConsumer
from netochi.pipeline.plotter import PipelinePlotterConfig
from netochi.pipeline.reporter import SummaryReporterConfig
from netochi.pipeline.runner.runner import (
    PipelineRunner,
    ExperimentTaskBase,
    ExperimentTaskRun, TaskBundle,
)
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle, BaselineStorer
from netochi.pipeline.metrics import ObjectiveConfigMetricConfig
from netochi.objectives.obj_log_likelihood import LogLikelihoodObjectiveConfig
from netochi.objectives.obj_inconsistency import InconsistencyObjectiveConfig, InconsistencyRelativeObjectiveConfig
from netochi.objectives.obj_hardware_size import MosaicHardwareSizeObjectiveConfig
from netochi.objectives.interfaces import MappingObjective
from netochi.pipeline.runner.baseline_provider import (
    RandomMosaicBaselineProviderConfig,
    MosaicGroundTruthBaselineProviderConfig
)

from netochi.pipeline.validator import ValidatorConfig

from netochi.visualization.visualize_adjacency_matrix import AdjacencyMatrixVisualizerConfig
from netochi.visualization.visualize_clustering import ClusteringVisualizerConfig
from netochi.visualization.visualize_mapping_output import MappingOutputVisualizerConfig
from netochi.visualization.visualize_routing_hierarchy import RoutingHierarchyVisualizerConfig

# ======================= CONFIGURE PIPELINE HERE =============================
PIPELINE_OUTPUT = PipelineOutputConfig().create()

HW_SMALL = MosaicHardwareConfig(
    nodes_per_router=2,
    neurons_per_core=15,
    router_levels=2,
    slice_factor=2
)

OBJECTIVE_CONFIGS = [
    InconsistencyObjectiveConfig(),
    InconsistencyRelativeObjectiveConfig(),
    MosaicHardwareSizeObjectiveConfig(),
    UnusedConnectionsObjectiveConfig(),
    LogLikelihoodObjectiveConfig()
]

HW_CONFIGS = [
    HW_SMALL
]

MAPPERS = [
    SimAnnealingIHWConfig(time_limit=2.0, pipeline_output=PIPELINE_OUTPUT).create(),
    #SimAnnealingMapperConfig().create(),
    #QAPPcaOptMapperConfig().create(),
    RandomMapperConfig().create(),
    #GreedyMapperConfig().create(),
    MosaicGroundTruthBaselineProviderConfig().create()
    #ILPMapperConfig().create(),
]

SEED = 42

HOOKS = [
    ValidatorConfig(pipeline_output=PIPELINE_OUTPUT).create()
]


CONSUMERS: List[PipelineConsumer[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]] = [
    SummaryReporterConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    PipelinePlotterConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    SummaryArchiverConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    AdjacencyMatrixVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    ClusteringVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    MappingOutputVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    RoutingHierarchyVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
]



# ==============================================================================

def define_task_inputs() -> BasePipelineRunner[MappingInput, MappingState, MappingState]:

    # 1. Define the inputs (Factories)
    probabilities = [0.1, 0.5]
    mosaic_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        MosaicNetworkConfig(hw_config=HW_SMALL, probability=p, seed=SEED).create() for p in probabilities
    ]
    er_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        ErdosRenyiConfig(hw_config=HW_SMALL, n=HW_SMALL.total_neurons, probability=p, seed=SEED).create() for p in probabilities
    ]
    swta_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        SwtaGeneratorConfig(hw_config=HW_SMALL,
                            num_clusters=10,
                            neurons_per_cluster=10,
                            inhibitory_ratio=0.2,
                            p_neighbor=0.1,
                            p_e_to_i=0.2,
                            p_i_to_e=0.8, seed=None).create()
    ]


    # 2. Define Baseline Providers
    random_baseline = RandomMosaicBaselineProviderConfig().create()
    gt_baseline = MosaicGroundTruthBaselineProviderConfig().create()
    metrics = [ObjectiveConfigMetricConfig(objective_config=obj_config).create() for obj_config in OBJECTIVE_CONFIGS]
    hooks: List[Tuple[MappingStateConsumer[BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]], Optional[BaselineStorer[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput]]]]] = [(hook, None) for hook in HOOKS]


    mosaic_evaluator_baseline_bundles = EvaluatorBundle(metrics_w_baselines=[(metric, gt_baseline) for metric in metrics], hooks=hooks)
    other_gen_evaluator_baseline_bundles = EvaluatorBundle(metrics_w_baselines=[(metric, random_baseline) for metric in metrics], hooks=hooks)

    mosaic_task_runs = [ExperimentTaskRun(evaluator_bundle=mosaic_evaluator_baseline_bundles,
                                          mapper=mapper) for mapper in MAPPERS]

    other_task_runs = [ExperimentTaskRun(evaluator_bundle=other_gen_evaluator_baseline_bundles,
                                         mapper=mapper) for mapper in MAPPERS]

    mosaic_tasks = ExperimentTaskBase(input_generators=mosaic_factories,
                                       evaluator_mapper_bundles=mosaic_task_runs,
                                       config=PIPELINE_OUTPUT)

    er_tasks = ExperimentTaskBase(input_generators=er_factories,
                                  evaluator_mapper_bundles=other_task_runs,
                                  config=PIPELINE_OUTPUT)

    swta_tasks = ExperimentTaskBase(input_generators=swta_factories,
                                    evaluator_mapper_bundles=other_task_runs,
                                    config=PIPELINE_OUTPUT)
    tasks: List[ExperimentTaskBase[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]] = [mosaic_tasks,
             #er_tasks,
             #swta_tasks
             ]

    bundles: List[TaskBundle[MappingInput, MappingState, MappingState]] = [TaskBundle(tasks=[task], consumer=CONSUMERS, config=PIPELINE_OUTPUT) for task in tasks]

    runner: BasePipelineRunner[MappingInput, MappingState, MappingState] = PipelineRunner(bundles=bundles, config=PIPELINE_OUTPUT)
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
