import time
from typing import List, Type
from netochi.pipeline.core import BaseInputFactory, BaseMapper, BaseMetric

class PipelineRunner:
    """Executes the benchmarking pipeline over factories, mappers, and metrics."""
    def __init__(
        self, 
        factories: List[BaseInputFactory],
        mapper_classes: List[Type[BaseMapper]],
        metric_classes: List[Type[BaseMetric]]
    ):
        """Initialize with factories, mapper classes, and metric classes."""
        self.factories = factories
        self.mapper_classes = mapper_classes
        self.metric_classes = metric_classes
        self.results = []
        
    def run(self):
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
                    except NotImplementedError:
                        print(f"  {mapper_name:<25} | {'SKIPPED (Unsupported Input)':>30} |")
                        continue
                    elapsed = time.time() - t0
                    
                    # Store the results for this execution
                    result_row = {
                        "mapper": mapper_name,
                        "time_s": elapsed,
                        **meta
                    }
                    
                    # Evaluate all metrics
                    for metric in metrics:
                        val = metric.evaluate(state)
                        result_row[metric.get_name()] = val
                        
                    self.results.append(result_row)
                    
                    # Print concise live feedback for this mapper
                    metrics_str = " | ".join([f"{m.get_name()}: {result_row[m.get_name()]:>10.2f}" for m in metrics])
                    print(f"  {mapper_name:<25} | {metrics_str} | {elapsed:.3f}s")
                    
        return self.results
        
    def print_results(self):
        """Detailed table is now optional or can be used for final summary."""
        pass
