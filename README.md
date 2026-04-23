# NeToChi
Network To Chip Framework for Institute for Neuroinformatics at ETH Zurich

## Quick Start

```bash
python3 -m pip install -r requirements.txt
```

```python
from src.network_generator.random_generator import RandomGenerator, RandomGeneratorConfig

config = RandomGeneratorConfig(
	nodes_per_router=2,
	neurons_per_core=8,
	router_levels=3,
	edge_probability=0.1,
)

result = RandomGenerator(seed=42).generate(config)
print(result.graph.number_of_nodes(), result.graph.number_of_edges())
print(result.assignment.shape)
```

