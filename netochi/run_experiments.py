from typing import List, Tuple, Optional, Union, Dict

from netochi.input_generator.erdos_renyi_factory import ErdosRenyiConfig
from netochi.input_generator.interfaces import MosaicMappingInput, MappingInput, HWBaseInputFactory
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
from netochi.input_generator.noisy_mosaic_network_factory import NoisyMosaicNetworkConfig
from netochi.input_generator.swta_factory import SwtaGeneratorConfig
from netochi.mapping.bottom_up_mapper import SliceSolveMode, ConstraintGraphMode
from netochi.mapping.feedback_loop_mapper import FeedbackLoopConfig
from netochi.mapping.interfaces import (
    BaseMosaicMappingState,
    MappingState
)
from netochi.mapping.random_mapper import RandomMapperConfig
from netochi.mapping.swta_mapper import SwtaMapperConfig
from netochi.mapping.swta_v2_mapper import SwtaV2MapperConfig
from netochi.mapping.theoretical_bounds_mapper import TheoreticalBoundsMapperConfig
from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import KaMinParHierarchicalClusterer
from netochi.mapping.three_step_mapping.clustering.clusterer.random_clusterer import RandomClusterer
from netochi.mapping.three_step_mapping.local_address_assignment.random_address_assigner import RandomAddressAssigner
from netochi.mapping.three_step_mapping.local_address_assignment.top_down_local_address_assigner import \
    TopDownLocalAddressAssignerConfig
from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner
from netochi.mapping.three_step_mapping.three_step_hw_mapper import ThreeStepHwMapperConfig
from netochi.objectives.obj_inconsistency import InconsistencyRelativeObjectiveConfig
from netochi.pipeline import BasePipelineRunner
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.enums import InputGraph
from netochi.pipeline.interfaces import PipelineConsumer, MappingStateConsumer
from netochi.pipeline.metrics import ObjectiveMetricConfig
from netochi.pipeline.pipeline_consumer.evaluation_csv_exporter_and_plotter import EvaluationExporterAndPlotterConfig
from netochi.pipeline.pipeline_consumer.reporter import SummaryReporterConfig
from netochi.pipeline.runner.baseline_provider import (
    RandomMosaicBaselineProviderConfig,
    MosaicGroundTruthBaselineProviderConfig
)
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle, BaselineStorer
from netochi.pipeline.runner.runner import (
    PipelineRunner,
    ExperimentTaskBase,
    ExperimentTaskRun, TaskBundle,
)
from netochi.pipeline.validator import ValidatorConfig

# ======================= CONFIGURE PIPELINE HERE =============================


# ---------- input configuration ------------
HW_3L_1 = MosaicHardwareConfig(neurons_per_core=16, router_levels=3) # 128 neurons
HW_3L_2 = MosaicHardwareConfig(neurons_per_core=32, router_levels=3) # 256 neurons
HW_3L_3 = MosaicHardwareConfig(neurons_per_core=64, router_levels=3) # 512 neurons
HW_3L_4 = MosaicHardwareConfig(neurons_per_core=128, router_levels=3) # 1024 neurons
HW_3L_5 = MosaicHardwareConfig(neurons_per_core=256, router_levels=3) # 2048 neurons

HW_4L_2 = MosaicHardwareConfig(neurons_per_core=16, router_levels=4) # 256 neurons
HW_4L_3 = MosaicHardwareConfig(neurons_per_core=32, router_levels=4) # 512 neurons
HW_4L_4 = MosaicHardwareConfig(neurons_per_core=64, router_levels=4) # 1024 neurons
HW_4L_5 = MosaicHardwareConfig(neurons_per_core=128, router_levels=4) # 2048 neurons

HW_CONFIGS = {
    HW_4L_5
    #HW_3L_1, HW_3L_2, HW_3L_3, HW_4L_4, HW_4L_5
}

SEEDS = [1,2,3,4,5] # [6,7,8,9,10]#

MOSAIC_PROBABILITIES = [0.5] # 1.0,
ER_PROBABILITIES = ["mosaic_p0.5"] # 0.05, 0.2,

INPUT_GRAPHS: List[InputGraph] = [
    InputGraph.ERDOS_RENYI,
    # InputGraph.MOSAIC,
    # InputGraph.NOISY_MOSAIC,
    # InputGraph.SWTA,
]
# ---------- mapper configuration ------------

MAPPERS = [
    SwtaV2MapperConfig().create(),
    #FeedbackLoopConfig(core_initializer=RandomClusterer(), address_initializer=RandomAddressAssigner()).create(),
    # ThreeStepHwMapperConfig(clusterer=KaMinParHierarchicalClusterer(),
    #                          address_assigner=TopDownLocalAddressAssignerConfig(
    #                              slice_solve_mode=SliceSolveMode.KL_CONSTRAINED_BISECTION,
    #                              constraint_graph_mode=ConstraintGraphMode.CLIQUE).create(),
    #                      slice_assigner=OptimalSliceAssigner()).create(),
    #TheoreticalBoundsMapperConfig().create(),
    #RandomMapperConfig().create(),

    # ThreeStepHwMapperConfig(clusterer=HierarchicalMetisClusterer(),
    #                         address_assigner=RandomAddressAssigner(),
    #                         slice_assigner=OptimalSliceAssigner()).create()

]

# ---------- output configuration ------------
OBJECTIVE_CONFIGS = [
    InconsistencyRelativeObjectiveConfig(),
    #CoreDistanceDifferenceObjectiveConfig(),
    #ClusteringAccuracyObjectiveConfig(),
    #InconsistencyObjectiveConfig(),
]

PIPELINE_OUTPUT = PipelineOutputConfig().create()

HOOKS = [
    ValidatorConfig(pipeline_output=PIPELINE_OUTPUT).create()
]

CONSUMERS: List[PipelineConsumer[MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput]]] = [
    SummaryReporterConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    EvaluationExporterAndPlotterConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #ClusteringVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
    #MappingOutputVisualizerConfig(pipeline_output=PIPELINE_OUTPUT).create(),
]



# ==============================================================================

def er_prob_matching_mosaic(hw_config, p_mosaic: float) -> float:
    """ER edge probability with the same expected number of edges as a
    Mosaic graph with probability p_mosaic on this hardware (binary routers)."""
    N = hw_config.total_neurons
    d_max = hw_config.router_levels
    C = 2 ** d_max                      # number of cores (binary routers)
    assert N % C == 0, "total_neurons must be divisible by the number of cores"
    n = N // C                          # neurons per core
    allowed_per_neuron = (n - 1) + d_max * n / 2
    return p_mosaic * allowed_per_neuron / (N - 1)

def resolve_er_prob(p: Union[float, str], hw_config) -> float:
    """Return p unchanged if it is a number. If it is "mosaic_p<value>",
    return the ER probability matching a Mosaic graph with that p."""
    if isinstance(p, str) and p.startswith("mosaic_p"):
        p_mosaic = float(p[len("mosaic_p"):])
        return er_prob_matching_mosaic(hw_config, p_mosaic)
    return float(p)

def define_task_inputs() -> BasePipelineRunner[MappingInput, MappingState, MappingState]:

    # 1. Define the inputs (Factories)
    # Inputs are (label, factory) pairs; here the label is the triple (probability, total_neurons, seed)
    mosaic_factories: List[Tuple[Tuple[float, int, int], HWBaseInputFactory[MosaicMappingInput]]] = [
        ((p, hw_config.total_neurons, seed), MosaicNetworkConfig(hw_config=hw_config, probability=p, seed=seed).create()) for p in MOSAIC_PROBABILITIES for hw_config in HW_CONFIGS for seed in SEEDS
    ]

    noisy_mosaic_factories: List[Tuple[Tuple[float, int, int], HWBaseInputFactory[MosaicMappingInput]]] = [
        ((p, hw_config.total_neurons, seed), NoisyMosaicNetworkConfig(hw_config=hw_config, probability=p, seed=seed, epsilon=0.1).create()) for p in MOSAIC_PROBABILITIES for hw_config in HW_CONFIGS for seed in SEEDS
    ]

    er_factories: List[Tuple[Tuple[float, int, int], HWBaseInputFactory[MosaicMappingInput]]] = [
        ((p, hw_config.total_neurons, seed), ErdosRenyiConfig(hw_config=hw_config, n=hw_config.total_neurons, probability=resolve_er_prob(p=p, hw_config=hw_config), seed=seed).create()) for p in ER_PROBABILITIES for hw_config in HW_CONFIGS for seed in SEEDS
    ]

    swta_factories: List[Tuple[Tuple[None, int, int], HWBaseInputFactory[MosaicMappingInput]]] = [
        ((None, hw_config.total_neurons, seed), SwtaGeneratorConfig(hw_config=hw_config,
                            num_clusters=hw_config.total_cores,
                            neurons_per_cluster=hw_config.neurons_per_core,
                            inhibitory_ratio=0.2,
                            p_neighbor=0.1,
                            p_e_to_i=0.2,
                            p_i_to_e=0.8,
                            seed=seed).create()) for seed in SEEDS
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
                                       config=PIPELINE_OUTPUT,
                                       label="mosaic")

    noisy_mosaic_tasks = ExperimentTaskBase(input_generators=noisy_mosaic_factories,
                                      evaluator_mapper_bundles=mosaic_task_runs,
                                      config=PIPELINE_OUTPUT,
                                      label="noisy_mosaic")

    er_tasks = ExperimentTaskBase(input_generators=er_factories,
                                  evaluator_mapper_bundles=other_task_runs,
                                  config=PIPELINE_OUTPUT,
                                  label="erdos_renyi")

    swta_tasks = ExperimentTaskBase(input_generators=swta_factories,
                                    evaluator_mapper_bundles=other_task_runs,
                                    config=PIPELINE_OUTPUT,
                                    label="swta")

    all_tasks: Dict[InputGraph, ExperimentTaskBase[
        MosaicMappingInput, BaseMosaicMappingState[MosaicMappingInput], BaseMosaicMappingState[MosaicMappingInput], str,
        Tuple[Optional[float], int, int]]] = {
        InputGraph.MOSAIC: mosaic_tasks,
        InputGraph.NOISY_MOSAIC: noisy_mosaic_tasks,
        InputGraph.ERDOS_RENYI: er_tasks,
        InputGraph.SWTA: swta_tasks,
    }
    tasks = [all_tasks[graph] for graph in dict.fromkeys(INPUT_GRAPHS)]  # keeps order, drops duplicates
    if not tasks:
        raise ValueError("INPUT_GRAPHS is empty: enable at least one input graph.")

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

    runner.run()


if __name__ == '__main__':
    run_experiment()
