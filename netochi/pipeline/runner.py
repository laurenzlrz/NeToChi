from netochi.input_generator import mosaic_hardware_config
from netochi.pipeline.exceptions import BaselineError
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generic, TypeVar, Any, Tuple, cast
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import MosaicNetworkMappingState, MosaicHWMappingState, BaseMosaicMappingState
from netochi.input_generator.interfaces import MosaicMappingInput

from netochi.pipeline.interfaces import MappingMetric, BasePipelineRunner
from netochi.pipeline.results import ExperimentResult, PipelineSummary
from netochi.pipeline.constants import (
    PIPELINE_LOG_FORMAT,
    MSG_TASK_HEADER,
    MSG_WITH_BASELINE,
    MSG_MAPPER_FAILED,
    KEY_GRAPH_TYPE,
    KEY_NODES,
    KEY_UNKNOWN,
    DEFAULT_REL_METRIC_VALUE,
    DEFAULT_METRIC_VALUE,
    REPORT_DIVIDER
)
from netochi.mapping.interfaces import BaseMapper, MappingState
from netochi.input_generator.interfaces import BaseInputFactory, MappingInput
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.storage import ResultManager
from netochi.pipeline.plotter import PipelinePlotter

PIPELINE_INPUT = TypeVar("PIPELINE_INPUT", bound=MappingInput[Any])
PIPELINE_INPUT_CONTRA = TypeVar("PIPELINE_INPUT_CONTRA", bound=MappingInput[Any], contravariant=True)
MAPPING_STATE = TypeVar("MAPPING_STATE", bound=MappingState[Any])
MAPPING_STATE_CO = TypeVar("MAPPING_STATE_CO", bound=MappingState[Any], covariant=True)
BASELINE_STATE = TypeVar("BASELINE_STATE", bound=MappingState[Any])
BASELINE_STATE_CO = TypeVar("BASELINE_STATE_CO", bound=MappingState[Any], covariant=True)


class Evaluator(BaseModel, Generic[MAPPING_STATE, BASELINE_STATE]):
    """Strongly typed container for metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    metrics: List[MappingMetric[MAPPING_STATE, BASELINE_STATE]]

    def evaluate_all(self, state: MAPPING_STATE, baseline: Optional[BASELINE_STATE]) -> Tuple[Dict[str, float], Dict[str, float]]:
        raw_results: Dict[str, float] = {}
        rel_results: Dict[str, float] = {}
        for metric in self.metrics:
            try:
                # Calculate Raw (Absolute)
                raw_results[metric.get_name()] = metric.evaluate(state)
                # Calculate Relative (if baseline exists)
                if baseline is not None:
                    rel_results[metric.get_name()] = metric.evaluate_against_baseline(state, baseline)
                else:
                    rel_results[metric.get_name()] = DEFAULT_METRIC_VALUE # No baseline
            except Exception as e:
                print(f"    Metric {metric.get_name()} failed: {e}")
        return raw_results, rel_results


class BaseBaselineProvider(ABC, Generic[BASELINE_STATE_CO, PIPELINE_INPUT_CONTRA]):
    """Base for generating baseline mapping states."""
    
    @abstractmethod
    def get_baseline(self, mapping_input: PIPELINE_INPUT_CONTRA) -> Optional[BASELINE_STATE_CO]:
        """Abstract baseline generation."""
        pass

    def get_name(self) -> str:
        """Returns a descriptive name for the baseline provider."""
        return self.__class__.__name__


class MosaicGroundTruthBaselineProvider(BaseBaselineProvider[BaseMosaicMappingState[MosaicMappingInput[Any]], MosaicMappingInput[Any]]):
    """Extracts ground truth from MosaicMappingInput or uses mapper to generate it"""
    
    def __init__(self, mapper: Optional[BaseMapper[BaseMosaicMappingState[Any], MosaicMappingInput[Any]]] = None) -> None:
        self.mapper = mapper

    def get_baseline(self, mapping_input: MosaicMappingInput[Any]) -> MosaicHWMappingState[MosaicMappingInput[Any], Any]:
        N = mapping_input.graph.num_vertices()
        hw = mapping_input.hw_config
        
        if mapping_input.neuron_core_pre_assignment is None:
            if self.mapper is None:
                raise BaselineError("No pre-assignment available for ground truth baseline and no mapper provided.")
            temp_state = self.mapper.run(mapping_input)
            # Ensure it's a HW state for the evaluator
            
            state = MosaicHWMappingState(
                mapping_input=mapping_input,
                hw_config=hw,
                neuron_core_idxs_assignment=temp_state.neuron_core_idxs_assignment,
                neuron_local_idxs_assignment=temp_state.neuron_local_idxs_assignment,
                neuron_slice_assignments=temp_state.neuron_slice_assignments
            )
        else:
            state = MosaicHWMappingState.from_input(mapping_input)
            for i in range(N):
                state.neuron_core_idxs_assignment[i] = i // hw.neurons_per_core
                state.neuron_local_idxs_assignment[i] = i % hw.neurons_per_core
            state.neuron_slice_assignments[:] = mapping_input.neuron_core_pre_assignment
            
        return state

    def get_name(self) -> str:
        if self.mapper is not None:
            return f"{self.mapper.get_name()}_MosaicGT"
        return "MosaicGT"


class MapperBaselineProvider(BaseBaselineProvider[MAPPING_STATE_CO, PIPELINE_INPUT_CONTRA], Generic[MAPPING_STATE_CO, PIPELINE_INPUT_CONTRA]):
    """Uses a specified mapper to generate a baseline state."""
    
    def __init__(self, mapper: BaseMapper[MAPPING_STATE_CO, PIPELINE_INPUT_CONTRA]) -> None:
        self.mapper = mapper

    def get_baseline(self, mapping_input: PIPELINE_INPUT_CONTRA) -> MAPPING_STATE_CO:
        return self.mapper.run(mapping_input)

    def get_name(self) -> str:
        return f"Baseline: {self.mapper.get_name()}"


class ExperimentTask(BaseModel, Generic[PIPELINE_INPUT, MAPPING_STATE, BASELINE_STATE]):
    """An execution step binding mappers, inputs, and baseline logic."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    mapper: BaseMapper[MAPPING_STATE, PIPELINE_INPUT]
    evaluator: Evaluator[MAPPING_STATE, BASELINE_STATE]
    
    # Each input factory is paired with a specific baseline provider
    inputs: List[Tuple[
        BaseInputFactory[PIPELINE_INPUT], 
        BaseBaselineProvider[BASELINE_STATE, PIPELINE_INPUT]
    ]]


class PipelineRunner(BaseModel, BasePipelineRunner):
    """
    Strictly typed pipeline runner using ExperimentTasks.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    tasks: List[ExperimentTask[MappingInput[Any], MappingState[Any], MappingState[Any]]]
    baselines: Dict[BaseInputFactory[Any], Tuple[BaseBaselineProvider[Any, Any], Evaluator[Any, Any]]] = Field(default_factory=dict)
    output_config: Optional[PipelineOutputConfig] = None
    verbose: bool = True

    def run(self) -> PipelineSummary:
        """Execute the pipeline across all configured tasks."""
        start_time = time.time()
        
        # 1. Run all tasks (Mappers)
        results = self._run_tasks()
        
        print("\n" + REPORT_DIVIDER)
        print("Running explicit baseline benchmarks...")
        print(REPORT_DIVIDER)

        # 2. Run explicit baseline benchmarks
        results.extend(self._run_baseline_benchmarks())

        print(REPORT_DIVIDER)
        print("Completed all tasks and baselines. Generating summary...")
        print(REPORT_DIVIDER)

        summary = PipelineSummary(
            results=results,
            total_time_s=time.time() - start_time
        )

        print(REPORT_DIVIDER)
        print("Summary completed successfully!")
        print(REPORT_DIVIDER)

        # 3. Handle persistence and plotting
        if self.output_config:
            self._save_and_plot(summary)
        
        print(REPORT_DIVIDER)
        print("Execution completed successfully!")
        print(REPORT_DIVIDER)
        
        return summary

    def _run_tasks(self) -> List[ExperimentResult]:
        """Runs the main mapping tasks."""
        results: List[ExperimentResult] = []
        for task in self.tasks:
            for factory, baseline_provider in task.inputs:
                mapping_input = factory.generate()
                meta = mapping_input.descriptions
                
                baseline = baseline_provider.get_baseline(mapping_input)

                if self.verbose:
                    baseline_info = MSG_WITH_BASELINE if baseline else ""
                    print(MSG_TASK_HEADER.format(
                        mapper=task.mapper.get_name(),
                        graph_type=meta.get(KEY_GRAPH_TYPE, KEY_UNKNOWN),
                        nodes=meta.get(KEY_NODES, KEY_UNKNOWN),
                        baseline_info=baseline_info
                    ))

                mapper_name = task.mapper.get_name()
                
                t0 = time.time()
                error_msg: Optional[str] = None
                state = None
                
                try:
                    state = task.mapper.run(mapping_input)
                except Exception as e:
                    error_msg = str(e)
                    if self.verbose:
                        print(f"  {mapper_name:<25} | FAILED: {error_msg}")
                
                elapsed = time.time() - t0
                
                raw_metric_values: Dict[str, float] = {}
                rel_metric_values: Dict[str, float] = {}
                if state:
                    raw_metric_values, rel_metric_values = task.evaluator.evaluate_all(state, baseline)
                else: 
                    print(MSG_MAPPER_FAILED)

                results.append(ExperimentResult(
                    mapper_name=mapper_name,
                    input_metadata=meta,
                    metrics=rel_metric_values,
                    raw_metrics=raw_metric_values,
                    execution_time_s=elapsed,
                    error=error_msg
                ))

                if self.verbose and not error_msg:
                    ll_val = next(iter(rel_metric_values.values())) if rel_metric_values else 0.0
                    print(PIPELINE_LOG_FORMAT.format(
                        mapper=mapper_name,
                        graph_type=meta.get(KEY_GRAPH_TYPE, KEY_UNKNOWN),
                        ll=ll_val,
                        elapsed=elapsed
                    ))
        return results

    def _run_baseline_benchmarks(self) -> List[ExperimentResult]:
        """Runs the explicit baseline benchmarks."""
        results: List[ExperimentResult] = []
        if not self.baselines:
            return results

        if self.verbose:
            print("\n" + "=" * 30)
            print("Running Baseline Benchmarks")
            print("=" * 30)
            
        for factory, (provider, evaluator) in self.baselines.items():
            mapping_input = factory.generate()
            meta = mapping_input.descriptions
            
            try:
                baseline_state = provider.get_baseline(mapping_input)
                if baseline_state:
                    raw_metrics, rel_metrics = evaluator.evaluate_all(baseline_state, baseline_state)
                        
                    results.append(ExperimentResult(
                        mapper_name=provider.get_name(),
                        input_metadata=meta,
                        metrics=rel_metrics,
                        raw_metrics=raw_metrics,
                        execution_time_s=0.0,
                        error=None
                    ))
                    if self.verbose:
                        print(f"  [Baseline] {meta.get(KEY_GRAPH_TYPE, KEY_UNKNOWN):<15} | Evaluated")
            except Exception as e:
                if self.verbose:
                    print(f"  [Baseline] {meta.get(KEY_GRAPH_TYPE, KEY_UNKNOWN):<15} | FAILED: {e}")
        return results

    def _find_evaluator_for_factory(self, factory: BaseInputFactory[Any]) -> Optional[Evaluator[Any, Any]]:
        """Finds an evaluator that is compatible with the given factory."""
        for task in self.tasks:
            for task_factory, _ in task.inputs:
                if task_factory == factory:
                    return task.evaluator
        if self.tasks:
            return self.tasks[0].evaluator
        return None

    def _save_and_plot(self, summary: PipelineSummary) -> None:
        """Saves results and generates plots."""
        if not self.output_config:
            return
            
        manager = ResultManager(config=self.output_config)
        run_dir = manager.save(summary)
        
        print(REPORT_DIVIDER)
        print("Saved results, Plotting results...")
        print(REPORT_DIVIDER)

        plotter = PipelinePlotter(config=self.output_config)
        plotter.plot_all(summary, run_dir)
        
        if self.verbose:
            print(f"\n[Storage] Results and plots saved to: {run_dir}")
