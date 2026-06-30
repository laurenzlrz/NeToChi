import time
from typing import List, Optional, Any, Sequence, Generic

from pydantic import BaseModel, ConfigDict, Field

from netochi.definitions.constants import NAME_OBJ_EXECUTION_TIME
from netochi.definitions.generics import Input_co, MappingState_co, BaselineState_co
from netochi.input_generator.interfaces import BaseInputFactory
from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import BaseMapper
from netochi.mapping.interfaces import MappingState
from netochi.pipeline.config import PipelineOutput
from netochi.pipeline.interfaces import BasePipelineRunner, PipelineConsumer
from netochi.pipeline.results import ExperimentResult, PipelineSummary
from netochi.pipeline.runner.evaluator_bundle import EvaluatorBundle


class ExperimentTaskRun[INPUT: MappingInput, MAPPING_STATE: MappingState, BASELINE_STATE: MappingState](BaseModel):
    """
    An execution step binding mappers, inputs, and baseline logic.
    Summarizes the combinatorical space and reduces repetitive input generations, when compatible:
    One input -> Multiple Mapper Runs and Baseline Runs -> Evaluations for each mapper
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)
    mapper: BaseMapper[MAPPING_STATE, INPUT] = Field(description="Mapper to execute for this task run.")
    evaluator_bundle: EvaluatorBundle[MAPPING_STATE, BASELINE_STATE, INPUT] = Field(description="Mapping of mappers to their corresponding evaluator bundles.")

    def run(self, input: INPUT) -> ExperimentResult:
        self.evaluator_bundle.prepare(input)
        assert isinstance(input, MappingInput), "Each input must be an instance of HWBaseInputFactory"
        assert isinstance(self.mapper, BaseMapper), "Each mapper must be an instance of BaseMapper"
        assert isinstance(self.evaluator_bundle, EvaluatorBundle), "Each evaluator bundle must be an instance of EvaluatorBundle"

        t0 = time.time()
        input_id = input.id
        state = self.mapper.run(input)
        elapsed = time.time() - t0
        raw_metrics, rel_metrics = self.evaluator_bundle.evaluate_all(state)
        raw_metrics[NAME_OBJ_EXECUTION_TIME] = elapsed

        error_msg: Optional[str] = None
        mapper_name = self.mapper.get_name()

        result = ExperimentResult(
            input_id=input_id,
            mapper_name=mapper_name,
            input_metadata=input.descriptions,
            metrics=rel_metrics,
            raw_metrics=raw_metrics,
            error=error_msg,
            state=state
        )
        return result


class ExperimentTaskBase(BaseModel, Generic[Input_co, MappingState_co, BaselineState_co]):
    """Base for experiment tasks, defining common attributes."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)

    # For convenience, we allow multiple input factories to be associated with the same mapper and baseline logic,
    # enabling combinatorial testing without needing to duplicate the entire task definition.
    input_generators: Sequence[BaseInputFactory[Input_co]] = Field(
        description="List of input factories to generate mapping inputs.")
    evaluator_mapper_bundles: Sequence[ExperimentTaskRun[Any, Any, Any]] = Field(
        description="Mapping of mappers to their corresponding evaluator bundles.")
    config: PipelineOutput = Field(description="Configuration for pipeline outputs.")

    def run(self) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []

        for input in self.input_generators:
            assert isinstance(input, BaseInputFactory), "Each input generator must be an instance of BaseInputFactory"
            input_instance = input.generate() # TODO Make Sure No Altering!
            input_name = input.get_name()
            self.config.print_console(f"Evaluating input generator: {input_name}")
            for evaluator_mapper_bundle in self.evaluator_mapper_bundles:
                mapper_name = evaluator_mapper_bundle.mapper.get_name()
                self.config.print_console(f"  --> Running mapper: {mapper_name}...")
                result = evaluator_mapper_bundle.run(input_instance)
                if result.error:
                    self.config.print_console(f"  --> Mapper failed: {result.error}")
                else:
                    self.config.print_console(f"  --> Mapper completed successfully in {result.raw_metrics[NAME_OBJ_EXECUTION_TIME]:.4f}s")
                results.append(result)

        return results


class TaskBundle(BaseModel, Generic[Input_co, MappingState_co, BaselineState_co]):
    """
    A bundle of tasks to be executed in a pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)

    tasks: Sequence[ExperimentTaskBase[Input_co, MappingState_co, BaselineState_co]] = Field(
        description="List of experiment tasks to execute in the pipeline.")
    consumer: Sequence[PipelineConsumer[Any, Any, Any]] = Field(
        description="List of consumers to process the pipeline summary after execution.")
    config: PipelineOutput = Field(description="Configuration for pipeline outputs.")

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


class PipelineRunner(BasePipelineRunner[Input_co, MappingState_co, BaselineState_co]):
    """
    Strictly typed pipeline runner using ExperimentTasks.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True)
    
    task_bundle: TaskBundle[Input_co, MappingState_co, BaselineState_co] = Field(description="Task bundle to execute in the pipeline.")

    def run(self) -> PipelineSummary:
        bundle_summary = self.task_bundle.run()
        return bundle_summary



