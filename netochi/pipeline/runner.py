import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generic, Any, Tuple
from pydantic import BaseModel, ConfigDict

from netochi.mapping.interfaces import MosaicNetworkMappingState, BaseMosaicMappingState, MappingState
from netochi.input_generator.interfaces import MosaicMappingInput, HWBaseInputFactory

from netochi.pipeline.interfaces import BasePipelineRunner
from netochi.pipeline.results import ExperimentResult, PipelineSummary
from netochi.definitions.constants import PIPELINE_LOG_FORMAT, MSG_TASK_HEADER, MSG_WITH_BASELINE, MSG_MAPPER_FAILED, \
    KEY_GRAPH_TYPE, KEY_NODES, KEY_UNKNOWN
from netochi.mapping.interfaces import BaseMapper
from netochi.input_generator.interfaces import MappingInput
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle
from tests.utils_mapping_output_validation import validate_mosaic_mapping


class BaseBaselineProvider(ABC, Generic[BASELINE_STATE_CO, PIPELINE_INPUT_CONTRA]):
    """Base for generating baseline mapping states."""
    
    @abstractmethod
    def get_baseline(self, mapping_input: PIPELINE_INPUT_CONTRA) -> Optional[BASELINE_STATE_CO]:
        """Abstract baseline generation."""
        pass


class MosaicGroundTruthBaselineProvider(BaseBaselineProvider[BaseMosaicMappingState, MappingInput]):
    """Extracts ground truth from MosaicMappingInput if available."""
    
    def get_baseline(self, mapping_input: MappingInput[Any]) -> Optional[BaseMosaicMappingState[Any]]:
            
        if not isinstance(mapping_input, MosaicMappingInput):
            return None
            
        try:
            state = MosaicNetworkMappingState.from_input(mapping_input)
            N = mapping_input.graph.num_vertices()
            hw = mapping_input.hw_config
            
            if mapping_input.pre_assignment is not None:
                for i in range(N):
                    state.neuron_core_idxs_assignment[i] = i // hw.neurons_per_core
                    state.neuron_local_idxs_assignment[i] = i % hw.neurons_per_core
                state.neuron_slice_assignments[:] = mapping_input.pre_assignment
                return state
        except Exception:
            pass
        return None


class MapperBaselineProvider(BaseBaselineProvider[MAPPING_STATE_CO, PIPELINE_INPUT_CONTRA], Generic[MAPPING_STATE_CO, PIPELINE_INPUT_CONTRA]):
    """Uses a specified mapper to generate a baseline state."""
    
    def __init__(self, mapper: BaseMapper[MAPPING_STATE_CO, PIPELINE_INPUT_CONTRA]) -> None:
        self.mapper = mapper

    def get_baseline(self, mapping_input: PIPELINE_INPUT_CONTRA) -> Optional[MAPPING_STATE_CO]:
        try:
            return self.mapper.run(mapping_input)
        except Exception:
            return None


class ExperimentTask[PIPELINE_INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE](BaseModel):
    """An execution step binding mappers, inputs, and baseline logic."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    mapper: BaseMapper[MAPPING_STATE, PIPELINE_INPUT]
    evaluator: EvaluatorBundle[MAPPING_STATE, BASELINE_STATE]
    
    # Each input factory is paired with a specific baseline provider
    inputs: List[Tuple[
        HWBaseInputFactory[PIPELINE_INPUT],
        BaseBaselineProvider[BASELINE_STATE, PIPELINE_INPUT]
    ]]


class PipelineRunner(BaseModel, BasePipelineRunner):
    """
    Strictly typed pipeline runner using ExperimentTasks.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    tasks: List[ExperimentTask[MappingInput, BaseMosaicMappingState, BaseMosaicMappingState]]
    verbose: bool = True

    def run(self) -> PipelineSummary:
        """Execute the pipeline across all configured tasks."""
        results: List[ExperimentResult] = []
        start_time = time.time()

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
                    if validate_mosaic_mapping(config=state.hw, state=state):
                        print("Output validated. \n")
                    raw_metric_values, rel_metric_values = task.evaluator.evaluate_all(state, baseline)
                else: 
                    print(MSG_MAPPER_FAILED)


                result = ExperimentResult(
                    mapper_name=mapper_name,
                    input_metadata=meta,
                    metrics=rel_metric_values,
                    raw_metrics=raw_metric_values,
                    execution_time_s=elapsed,
                    error=error_msg
                )
                results.append(result)

                if self.verbose and not error_msg:
                    ll_val = next(iter(rel_metric_values.values())) if rel_metric_values else 0.0
                    print(PIPELINE_LOG_FORMAT.format(
                        mapper=mapper_name,
                        graph_type=meta.get(KEY_GRAPH_TYPE, KEY_UNKNOWN),
                        ll=ll_val,
                        elapsed=elapsed
                    ))

        return PipelineSummary(
            results=results,
            total_time_s=time.time() - start_time
        )
