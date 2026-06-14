import time
from typing import List, Dict, Optional, Any, Sequence
from pydantic import BaseModel, ConfigDict, Field

from netochi.mapping.interfaces import MappingState
from netochi.input_generator.interfaces import BaseInputFactory

from netochi.pipeline.interfaces import BasePipelineRunner
from netochi.pipeline.pipeline_consumer import PipelineConsumer
from netochi.pipeline.results import ExperimentResult, PipelineSummary
from netochi.mapping.interfaces import BaseMapper
from netochi.input_generator.interfaces import MappingInput
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle
from netochi.pipeline.runner.evaluator_bundle import BaselineStorer


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
    def baseline_storers(self) -> List[BaselineStorer[INPUT, BASELINE_STATE]]:
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
    input_generators: Sequence[BaseInputFactory[INPUT]] = Field(
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


class TaskBundle(BaseModel):
    """
    A bundle of tasks to be executed in a pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)

    tasks: List[ExperimentTaskBase[Any]] = Field(
        description="List of experiment tasks to execute in the pipeline.")
    consumer: List[PipelineConsumer] = Field(
        description="List of consumers to process the pipeline summary after execution.")

    def run(self):
        results = []
        start_time = time.time()
        for task in self.tasks:
            task_results = task.run()
            results.extend(task_results)
        end_time = time.time()
        time_delta = end_time - start_time
        pipeline_summary = PipelineSummary(results=results, total_time_s=time_delta)
        for consumer in self.consumer:
            consumer.consume(pipeline_summary)
        return pipeline_summary


class PipelineRunner(BasePipelineRunner):
    """
    Strictly typed pipeline runner using ExperimentTasks.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)
    
    bundles: List[TaskBundle] = Field(description="List of task bundles to execute in the pipeline.")

    def run(self) -> List[PipelineSummary]:
        overall_results = []
        for bundle in self.bundles:
            bundle_summary = bundle.run()
            overall_results.append(bundle_summary)
        return overall_results



