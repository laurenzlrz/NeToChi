import time
from typing import List, Dict, Optional, Generic, TypeVar
from pydantic import BaseModel, ConfigDict

from netochi.pipeline.interfaces import BasePipelineRunner, MappingMetric
from netochi.pipeline.results import ExperimentResult, PipelineSummary
from netochi.pipeline.constants import PIPELINE_LOG_FORMAT
from netochi.mapping.interfaces import (
    BaseMapper, 
    MappingState, 
    ANY_MAPPING_INPUT,
    MosaicNetworkMappingState
)
from netochi.input_generator.interfaces import BaseInputFactory, MosaicMappingInput


MAPPING_STATE = TypeVar("MAPPING_STATE", bound=MappingState)


class PipelineRunner(BaseModel, BasePipelineRunner, Generic[ANY_MAPPING_INPUT, MAPPING_STATE]):
    """
    Pydantic-based benchmark runner.
    Coordinates factories, mappers, and metrics to produce structured results.
    Supports baseline-aware evaluation if input provides reference assignments.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    factories: List[BaseInputFactory[ANY_MAPPING_INPUT]]
    mappers: List[BaseMapper[MAPPING_STATE, ANY_MAPPING_INPUT]]
    metrics: List[MappingMetric[MAPPING_STATE, MAPPING_STATE]]
    verbose: bool = True

    def run(self) -> PipelineSummary:
        """Execute the pipeline Cartesian product."""
        results: List[ExperimentResult] = []
        start_time = time.time()

        for factory in self.factories:
            mapping_input = factory.generate()
            meta = mapping_input.descriptions
            
            # Attempt to create a baseline state for comparative evaluation
            baseline: Optional[MAPPING_STATE] = self._create_baseline_state(mapping_input)
            
            if self.verbose:
                baseline_info = " (with baseline)" if baseline else ""
                print(f"\n--- Input: {meta.get('graph_type')} (Nodes={meta.get('nodes')}, Edges={meta.get('edges')}){baseline_info} ---")

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
                            # Evaluate with optional baseline
                            val = metric.evaluate(state, baseline=baseline)
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

    def _create_baseline_state(self, mapping_input: ANY_MAPPING_INPUT) -> Optional[MAPPING_STATE]:
        """Creates a baseline MappingState if the input contains pre-assignment data."""
        if isinstance(mapping_input, MosaicMappingInput) and mapping_input.pre_assignment is not None:
            # For Mosaic inputs with pre-assignments, create a MosaicNetworkMappingState
            # and populate it with the ground truth.
            try:
                state = MosaicNetworkMappingState.from_input(mapping_input)
                # pre_assignment is expected to be the slice assignments s[neuron, dist]
                # We assume c and x can be derived from indices if not provided
                N = mapping_input.graph.num_vertices()
                hw = mapping_input.hw_config
                for i in range(N):
                    state.neuron_core_idxs_assignment[i] = i // hw.neurons_per_core
                    state.neuron_local_idxs_assignment[i] = i % hw.neurons_per_core
                
                state.neuron_slice_assignments[:] = mapping_input.pre_assignment
                return state  # type: ignore
            except Exception:
                return None
        return None
