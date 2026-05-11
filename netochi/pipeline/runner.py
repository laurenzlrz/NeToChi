import time
from typing import List, Dict, Optional, Generic, TypeVar
from pydantic import BaseModel, ConfigDict

from netochi.pipeline.interfaces import BasePipelineRunner, MappingMetric
from netochi.pipeline.results import ExperimentResult, PipelineSummary
from netochi.pipeline.constants import PIPELINE_LOG_FORMAT
from netochi.mapping.interfaces import BaseMapper, MappingState, ANY_MAPPING_INPUT
from netochi.input_generator.interfaces import BaseInputFactory


MAPPING_STATE = TypeVar("MAPPING_STATE", bound=MappingState)


class PipelineRunner(BaseModel, BasePipelineRunner, Generic[ANY_MAPPING_INPUT, MAPPING_STATE]):
    """
    Pydantic-based benchmark runner.
    Coordinates factories, mappers, and metrics to produce structured results.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    factories: List[BaseInputFactory[ANY_MAPPING_INPUT]]
    mappers: List[BaseMapper[MAPPING_STATE, ANY_MAPPING_INPUT]]
    metrics: List[MappingMetric[MAPPING_STATE]]
    verbose: bool = True

    def run(self) -> PipelineSummary:
        """Execute the pipeline Cartesian product."""
        results: List[ExperimentResult] = []
        start_time = time.time()

        for factory in self.factories:
            # Note: Current factories return a single MappingInput
            mapping_input = factory.generate()
            meta = mapping_input.descriptions
            
            if self.verbose:
                print(f"\n--- Input: {meta.get('graph_type')} (Nodes={meta.get('nodes')}, Edges={meta.get('edges')}) ---")

            for mapper in self.mappers:
                mapper_name = mapper.get_name()
                
                t0 = time.time()
                error_msg: Optional[str] = None
                state: Optional[MAPPING_STATE] = None
                
                try:
                    state = mapper.run(mapping_input)
                except Exception as e:
                    error_msg = str(e)
                    if self.verbose:
                        print(f"  {mapper_name:<25} | FAILED: {error_msg}")
                
                elapsed = time.time() - t0
                
                metric_values: Dict[str, float] = {}
                if state:
                    for metric in self.metrics:
                        try:
                            val = metric.evaluate(state)
                            metric_values[metric.get_name()] = val
                        except Exception as e:
                            print(f"    Metric {metric.get_name()} failed: {e}")
                
                result = ExperimentResult(
                    mapper_name=mapper_name,
                    input_metadata=meta,
                    metrics=metric_values,
                    execution_time_s=elapsed,
                    error=error_msg
                )
                results.append(result)

                if self.verbose and not error_msg:
                    # Log first metric value as a placeholder for LL in the old format
                    ll_val = next(iter(metric_values.values())) if metric_values else 0.0
                    print(PIPELINE_LOG_FORMAT.format(
                        mapper=mapper_name,
                        graph_type=meta.get('graph_type', 'Unknown'),
                        ll=ll_val,
                        elapsed=elapsed
                    ))

        return PipelineSummary(
            results=results,
            total_time_s=time.time() - start_time
        )
