import time
from typing import List, Type, Dict, Any
from pydantic import BaseModel, Field
from netochi.pipeline.core import InputFactory, Mapper, Metric, MappingInput
from netochi.mapping.likelihood_state import MappingResult

class ExperimentResult(BaseModel):
    """Result of a single experiment run."""
    mapper: str
    time_s: float
    metadata: Dict[str, Any]
    metrics: Dict[str, float]

class PipelineRunner:
    """Executes the benchmarking pipeline over factories, mappers, and metrics."""
    def __init__(
        self, 
        factories: List[InputFactory],
        mapper_classes: List[Type[Mapper]],
        metric_classes: List[Type[Metric]]
    ):
        """Initialize with factories, mapper classes, and metric classes."""
        self.factories = factories
        self.mapper_classes = mapper_classes
        self.metric_classes = metric_classes
        self.results: List[ExperimentResult] = []
        
    def run(self) -> List[ExperimentResult]:
        """Execute the pipeline Cartesian product and collect results."""
        metrics = [m() for m in self.metric_classes]
        
        for factory in self.factories:
            for mapping_input, meta in factory.generate():
                # Print input header
                print(f"\n--- Input: {meta.get('graph_type')} (Nodes={meta.get('nodes')}, Edges={meta.get('edges')}, Prob={meta.get('edge_prob')}) ---")
                
                # For each generated input, evaluate all mappers
                for mapper_cls in self.mapper_classes:
                    mapper = mapper_cls()
                    mapper_name = mapper.get_name()
                    
                    t0 = time.time()
                    try:
                        state = mapping_input.accept(mapper)
                    except (NotImplementedError, AttributeError) as e:
                        print(f"  {mapper_name:<25} | {'SKIPPED (Unsupported Input)':>30} |")
                        continue
                        
                    elapsed = time.time() - t0
                    
                    # Evaluate all metrics
                    metric_values = {}
                    for metric in metrics:
                        val = metric.evaluate(state)
                        metric_values[metric.get_name()] = val
                        
                    result = ExperimentResult(
                        mapper=mapper_name,
                        time_s=elapsed,
                        metadata=meta,
                        metrics=metric_values
                    )
                    
                    self.results.append(result)
                    
                    # Print concise live feedback for this mapper
                    metrics_str = " | ".join([f"{name}: {val:>10.2f}" for name, val in metric_values.items()])
                    print(f"  {mapper_name:<25} | {metrics_str} | {elapsed:.3f}s")
                    
        return self.results
        
    def print_results(self):
        """Final summary of results."""
        print("\n" + "="*80)
        print(f"{'Mapper':<25} | {'Graph':<20} | {'Log-Likelihood':<15} | {'Time (s)':<10}")
        print("-" * 80)
        for res in self.results:
            ll = res.metrics.get("LikelihoodObjective", 0.0)
            graph_type = res.metadata.get("graph_type", "Unknown")
            print(f"{res.mapper:<25} | {graph_type:<20} | {ll:<15.2f} | {res.time_s:<10.3f}")
        print("="*80)
