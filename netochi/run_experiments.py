from typing import List, Tuple, Optional

from networkx.algorithms.bipartite import clustering

from netochi.input_generator.noisy_mosaic_network_factory import NoisyMosaicNetworkConfig
from netochi.mapping import SimAnnealingIHWConfig
from netochi.mapping.bottom_up_mapper import BottomUpMapperConfig, SliceSolveMode, ConstraintGraphMode
from netochi.mapping.feedback_loop_mapper import FeedbackLoopConfig
from netochi.mapping.ilp_mapper import ILPMapper, ILPMapperConfig
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
    MappingState
)
from netochi.input_generator.interfaces import MosaicMappingInput, MappingInput, HWBaseInputFactory

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.erdos_renyi_factory import ErdosRenyiConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
from netochi.input_generator.swta_factory import SwtaGeneratorConfig

from netochi.mapping.greedy_mapper import GreedyMapperConfig
from netochi.mapping.random_mapper import RandomMapperConfig
from netochi.mapping.simulated_annealing_mapper import SimAnnealingMapperConfig
from netochi.mapping.simulated_annealing_mapper_cluster_init import SimulatedAnnealingMapperClusterInit, \
    SimulatedAnnealingMapperClusterInitConfig
from netochi.mapping.theoretical_bounds_mapper import TheoreticalBoundsMapperConfig
from netochi.mapping.three_step_mapping.clustering.clusterer.bottom_up_clusterer import BottomUpClusterer
from netochi.mapping.three_step_mapping.clustering.clusterer.gt_clusterer import GroundTruthOrRandomClusterer
from netochi.mapping.three_step_mapping.clustering.clusterer.kaHyPar_cluster import KaHyParHierarchicalClusterer, \
    kaHyParClusterFunction
from netochi.mapping.three_step_mapping.clustering.clusterer.kaHyPar_hyperedges_clusterer import \
    KaHyParHyperedgesClusterer, kaHyParHyperedgesClusterFunction
from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import KaMinParHierarchicalClusterer, \
    kaMinParClusterFunction
from netochi.mapping.three_step_mapping.clustering.clusterer.spectral_tree_clusterer import SpectralTreeClusterer
from netochi.mapping.three_step_mapping.clustering.hcd_hw_clusterer import HcdHwClusterer
from netochi.mapping.three_step_mapping.clustering.hsbm_hw_clusterer import HsbmHwClusterer
from netochi.mapping.three_step_mapping.clustering.qap_hw_clusterer import QapHwClusterer
from netochi.mapping.three_step_mapping.local_address_assignment.cofanin_address_assigner import CoFanInAddressAssigner
from netochi.mapping.three_step_mapping.local_address_assignment.gt_local_assigner import \
    GroundTruthOrRandomLocalAddressAssigner
from netochi.mapping.three_step_mapping.local_address_assignment.lp_local_address_assigner import PipageRoundingAssigner
from netochi.mapping.three_step_mapping.local_address_assignment.pca_local_address_assigner import \
    PcaLocalAddressAssigner
from netochi.mapping.three_step_mapping.local_address_assignment.sdp_local_address_assigner import SDPRoundingAssigner
from netochi.mapping.three_step_mapping.local_address_assignment.sim_annealing_local_address_assigner import SALocalAddressAssignerConfig
from netochi.mapping.three_step_mapping.local_address_assignment.top_down_local_address_assigner import \
    TopDownLocalAddressAssignerConfig
from netochi.mapping.three_step_mapping.qap_pca_opt_three_step_mapper import QAPPcaOptMapperConfig
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner
from netochi.mapping.three_step_mapping.three_step_hw_mapper import ThreeStepHwMapperConfig
from netochi.mapping.three_step_mapping.three_step_mapper import ThreeStepMapperConfig
from netochi.objectives.obj_clustering import ClusteringARIObjective, ClusteringARIObjectiveConfig, \
    ClusteringAccuracyObjectiveConfig, CoreDistanceDifferenceObjectiveConfig

from netochi.objectives.obj_unused_connections import UnusedConnectionsObjectiveConfig
from netochi.pipeline import BasePipelineRunner, PipelineSummary
from netochi.pipeline.pipeline_consumer.archiver import SummaryArchiverConfig
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.interfaces import PipelineConsumer, MappingStateConsumer
from netochi.pipeline.pipeline_consumer.evaluation_csv_exporter import EvaluationExporter, EvaluationExporterConfig
from netochi.pipeline.pipeline_consumer.evaluation_csv_exporter_and_plotter import EvaluationExporterAndPlotterConfig
from netochi.pipeline.pipeline_consumer.plotter import PipelinePlotterConfig
from netochi.pipeline.pipeline_consumer.reporter import SummaryReporterConfig
from netochi.pipeline.runner.runner import (
    PipelineRunner,
    ExperimentTaskBase,
    ExperimentTaskRun, TaskBundle,
)
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle, BaselineStorer
from netochi.pipeline.metrics import ObjectiveMetricConfig
from netochi.objectives.obj_log_likelihood import LogLikelihoodObjectiveConfig
from netochi.objectives.obj_inconsistency import InconsistencyObjectiveConfig, InconsistencyRelativeObjectiveConfig
from netochi.objectives.obj_hardware_size import MosaicHardwareSizeObjectiveConfig
from netochi.pipeline.runner.baseline_provider import (
    RandomMosaicBaselineProviderConfig,
    MosaicGroundTruthBaselineProviderConfig
)

from netochi.pipeline.validator import ValidatorConfig

from netochi.visualization.visualize_adjacency_matrix import AdjacencyMatrixVisualizerConfig
from netochi.visualization.visualize_clustering import ClusteringVisualizerConfig
from netochi.visualization.visualize_mapping_output import MappingOutputVisualizerConfig
from netochi.visualization.visualize_routing_hierarchy import RoutingHierarchyVisualizerConfig

from netochi.pipeline.pipeline_consumer.event_processor import EventProcessor, EventProcessorConfig
from netochi.pipeline.pipeline_consumer.objective_consumer import CustomObjectiveConsumer

# ======================= CONFIGURE PIPELINE HERE =============================
PIPELINE_OUTPUT = PipelineOutputConfig().create()

HW_MINI = MosaicHardwareConfig(
    nodes_per_router=2,
    neurons_per_core=4,
    router_levels=2,
    slice_factor=2
)

HW_SMALL = MosaicHardwareConfig(
    nodes_per_router=2,
    neurons_per_core=16,
    router_levels=2,
    slice_factor=2
)

HW_SMALL_TRINARY = MosaicHardwareConfig(
    nodes_per_router=3,
    neurons_per_core=16,
    router_levels=2,
    slice_factor=2
)

HW_MEDIUM = MosaicHardwareConfig(
    nodes_per_router=2,
    neurons_per_core=16,
    router_levels=3,
    slice_factor=2
)

HW_MEDIUM_TRINARY = MosaicHardwareConfig(
    nodes_per_router=3,
    neurons_per_core=16,
    router_levels=3,
    slice_factor=2
)

HW_LARGE = MosaicHardwareConfig(
    nodes_per_router=2,
    neurons_per_core=32,
    router_levels=3,
    slice_factor=2
)

HW_LARGE_2 = MosaicHardwareConfig(
    nodes_per_router=2,
    neurons_per_core=64,
    router_levels=3,
    slice_factor=2
)

HW_LARGE_TRINARY = MosaicHardwareConfig(
    nodes_per_router=3,
    neurons_per_core=32,
    router_levels=3,
    slice_factor=2
)


OBJECTIVE_CONFIGS = [
    InconsistencyRelativeObjectiveConfig(),
    #ClusteringARIObjectiveConfig(),
    #ClusteringAccuracyObjectiveConfig(),
    #CoreDistanceDifferenceObjectiveConfig(),
    #InconsistencyObjectiveConfig(),
    #MosaicHardwareSizeObjectiveConfig(),
    #UnusedConnectionsObjectiveConfig(),
    #MosaicHardwareSizeObjectiveConfig(),
    #UnusedConnectionsObjectiveConfig(),
    #LogLikelihoodObjectiveConfig()
]

HW_CONFIGS = [
    #HW_MEDIUM,
    #HW_LARGE_2,

    #HW_MINI,
    HW_SMALL,
    #HW_SMALL_TRINARY,
    ##HW_MEDIUM_TRINARY,
    HW_LARGE,
    #HW_LARGE_TRINARY
]

SEEDS = [1,2] #3,4,5]

MAPPERS = [
    #ThreeStepHwMapperConfig(clusterer=BottomUpClusterer(clusterer=kaMinParClusterFunction), address_assigner=PipageRoundingAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=BottomUpClusterer(clusterer=kaMinParClusterFunction), address_assigner=SALocalAddressAssignerConfig().create(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=BottomUpClusterer(clusterer=kaMinParClusterFunction), address_assigner=TopDownLocalAddressAssignerConfig(slice_solve_mode=SliceSolveMode.KL_CONSTRAINED_BISECTION, constraint_graph_mode=ConstraintGraphMode.CLIQUE).create(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=QapHwClusterer(), address_assigner=TopDownLocalAddressAssignerConfig(slice_solve_mode=SliceSolveMode.KL_CONSTRAINED_BISECTION, constraint_graph_mode=ConstraintGraphMode.CLIQUE).create(), slice_assigner=OptimalSliceAssigner()).create(),
    RandomMapperConfig().create(),
    #FeedbackLoopConfig().create(),
    #ThreeStepHwMapperConfig(clusterer=SpectralTreeClusterer(), address_assigner=CoFanInAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),

    #SimulatedAnnealingMapperClusterInitConfig().create(),
    # SDP-based recovery mapper (TheoreticalBoundsProof/alg1). Only for binary routers, slice_factor=2,
    # power-of-two neurons_per_core and |V| = total_neurons (raises otherwise, e.g. on *_TRINARY configs).
    # The core SDP has N^2 variables (N = 512, HW_LARGE_2: ~7 s with SCS); refused above max_neurons (default 1024).
    TheoreticalBoundsMapperConfig().create(),
    #ILPMapperConfig().create()

    #ThreeStepHwMapperConfig(clusterer=BottomUpClusterer(clusterer=kaHyParHyperedgesClusterFunction), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=BottomUpClusterer(clusterer=kaHyParClusterFunction), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=KaMinParHierarchicalClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=KaHyParHyperedgesClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=KaHyParHierarchicalClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=HsbmHwClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=HcdHwClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=QapHwClusterer(), address_assigner=PcaLocalAddressAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=HsbmHwClusterer(), address_assigner=PipageRoundingAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=HcdHwClusterer(), address_assigner=PipageRoundingAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
    #ThreeStepHwMapperConfig(clusterer=QapHwClusterer(), address_assigner=PipageRoundingAssigner(), slice_assigner=OptimalSliceAssigner()).create(),
]


PROBABILITIES = [0.5, 1.0]

HOOKS = [
    ValidatorConfig(pipeline_output=PIPELINE_OUTPUT).create()
]


CONSUMERS: List[PipelineConsumer[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]] = [
    SummaryReporterConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    EvaluationExporterAndPlotterConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #ClusteringVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #MappingOutputVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),

    #PipelinePlotterConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #SummaryArchiverConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #AdjacencyMatrixVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #RoutingHierarchyVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #EventProcessor(EventProcessorConfig(objective_consumer=[CustomObjectiveConsumer("MyCustomObjectiveConsumer")]))
]



# ==============================================================================

def define_task_inputs() -> BasePipelineRunner[MappingInput, MappingState, MappingState]:

    # 1. Define the inputs (Factories)
    mosaic_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        MosaicNetworkConfig(hw_config=hw_config, probability=p, seed=seed).create() for p in PROBABILITIES for hw_config in HW_CONFIGS for seed in SEEDS
    ]

    noisy_mosaic_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        NoisyMosaicNetworkConfig(hw_config=hw_config, probability=p, seed=seed, epsilon=0.1).create() for p in PROBABILITIES for hw_config in HW_CONFIGS for seed in SEEDS
    ]

    er_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        ErdosRenyiConfig(hw_config=hw_config, n=hw_config.total_neurons, probability=p, seed=seed).create() for p in PROBABILITIES for hw_config in HW_CONFIGS for seed in SEEDS
    ]
    swta_factories: List[HWBaseInputFactory[MosaicMappingInput]] = [
        SwtaGeneratorConfig(hw_config=hw_config,
                            num_clusters=hw_config.total_cores,
                            neurons_per_cluster=hw_config.neurons_per_core,
                            inhibitory_ratio=0.2,
                            p_neighbor=0.1,
                            p_e_to_i=0.2,
                            p_i_to_e=0.8,
                            seed=seed).create() for seed in SEEDS
        for hw_config in HW_CONFIGS
    ]


    # 2. Define Baseline Providers
    random_baseline = RandomMosaicBaselineProviderConfig().create()
    gt_baseline = MosaicGroundTruthBaselineProviderConfig().create()
    metrics = [ObjectiveMetricConfig(objective_config=obj_config).create() for obj_config in OBJECTIVE_CONFIGS]
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

    noisy_mosaic_tasks = ExperimentTaskBase(input_generators=noisy_mosaic_factories,
                                      evaluator_mapper_bundles=mosaic_task_runs,
                                      config=PIPELINE_OUTPUT)

    er_tasks = ExperimentTaskBase(input_generators=er_factories,
                                  evaluator_mapper_bundles=other_task_runs,
                                  config=PIPELINE_OUTPUT)

    swta_tasks = ExperimentTaskBase(input_generators=swta_factories,
                                    evaluator_mapper_bundles=other_task_runs,
                                    config=PIPELINE_OUTPUT)
    tasks: List[ExperimentTaskBase[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]] = [
        mosaic_tasks,
        #er_tasks,
        noisy_mosaic_tasks,
        swta_tasks
    ]

    task_bundle: TaskBundle[MappingInput, MappingState, MappingState] = TaskBundle(tasks=tasks, consumer=CONSUMERS, config=PIPELINE_OUTPUT)

    runner: BasePipelineRunner[MappingInput, MappingState, MappingState] = PipelineRunner(task_bundle=task_bundle, config=PIPELINE_OUTPUT)
    return runner


def run_experiment() -> None:

    # === 1. define mosaic tasks ===
    runner = define_task_inputs()

    # === 2. Run Pipeline ===
    print("=" * 100)
    print("Neuromorphic Mapping Pipeline Execution")
    print("=" * 100)

    summary = runner.run()


if __name__ == '__main__':
    run_experiment()
