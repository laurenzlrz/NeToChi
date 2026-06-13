import time
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field

from config import PipelineOutputConfig
from netochi.mapping.interfaces import MappingState
from netochi.input_generator.interfaces import BaseInputFactory

from netochi.pipeline.interfaces import BasePipelineRunner
from netochi.pipeline.results import ExperimentResult, PipelineSummary
from netochi.mapping.interfaces import BaseMapper
from netochi.input_generator.interfaces import MappingInput
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle
from runner import baseline_provider
from runner.evaluator_bundle import BaselineStorer


class ExperimentTaskRun[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel):
    """
    An execution step binding mappers, inputs, and baseline logic.
    Summarizes the combinatorical space and reduces repetitive input generations, when compatible:
    One input -> Multiple Mapper Runs and Baseline Runs -> Evaluations for each mapper
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)
    mapper: BaseMapper[MAPPING_STATE, INPUT] = Field(description="Mapper to execute for this task run.")
    evaluator_bundle: EvaluatorBundle[MAPPING_STATE, BASELINE_STATE, INPUT] = Field(description="Mapping of mappers to their corresponding evaluator bundles.")

    #TODO

    @property
    def baseline_storers(self) -> List[BaselineStorer[BASELINE_STATE, INPUT]]:
        return self.evaluator_bundle.get_baselines

    def run(self, input: INPUT) -> ExperimentResult:
        assert isinstance(input, MappingInput), "Each input must be an instance of HWBaseInputFactory"

        assert isinstance(self.mapper, BaseMapper), "Each mapper must be an instance of BaseMapper"
        assert isinstance(self.evaluator_bundle, EvaluatorBundle), "Each evaluator bundle must be an instance of EvaluatorBundle"

        t0 = time.time()
        state = self.mapper.run(input)
        elapsed = time.time() - t0
        raw_metrics, rel_metrics = self.evaluator_bundle.evaluate_all(state)

        error_msg: Optional[str] = None
        mapper_name = self.mapper.get_name()

        result = ExperimentResult(
            mapper_name=mapper_name,
            input_metadata=input.descriptions,
            metrics=rel_metrics,
            raw_metrics=raw_metrics,
            execution_time_s=elapsed,
            error=error_msg
        )
        return result


class ExperimentTaskBase[INPUT: MappingInput](BaseModel):
    """Base for experiment tasks, defining common attributes."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)

    # For convenience, we allow multiple input factories to be associated with the same mapper and baseline logic,
    # enabling combinatorial testing without needing to duplicate the entire task definition.
    input_generators: List[BaseInputFactory[INPUT]] = Field(
        description="List of input factories to generate mapping inputs.")
    evaluator_mapper_bundles: List[ExperimentTaskRun[INPUT, Any, Any]] = Field(
        description="Mapping of mappers to their corresponding evaluator bundles.")

    def run(self) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []

        baselines_runner_list: List[BaselineStorer[INPUT, Any]] = [
            storer
            for evaluator_bundle in self.evaluator_mapper_bundles
            for storer in evaluator_bundle.baseline_storers
        ]

        for input in self.input_generators:
            assert isinstance(input, BaseInputFactory), "Each input generator must be an instance of BaseInputFactory"
            input_instance = input.generate() # TODO Make Sure No Altering!
            for storer in baselines_runner_list:
                storer.run(input_instance)
            for evaluator_mapper_bundle in self.evaluator_mapper_bundles:
                result = evaluator_mapper_bundle.run(input_instance)
                results.append(result)

        return results


class PipelineRunner(BaseModel, BasePipelineRunner):
    """
    Strictly typed pipeline runner using ExperimentTasks.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)
    
    tasks: List[ExperimentTaskBase[Any]]
    context_config = PipelineOutputConfig()

    def run(self) -> PipelineSummary:
        """Execute the pipeline across all configured tasks."""
        results: List[ExperimentResult] = []
        start_time = time.time()

        for task in self.tasks:
            task_results = task.run()
            results.extend(task_results)

        pipeline_summary = PipelineSummary(
            results=results,
            total_time_s=time.time() - start_time
        )

        return pipeline_summary



