# NeToChi 🕸️🚀

**Neuromorphic To Chip: Stochastic Block Model Mapping Framework**

NeToChi is a high-performance framework designed to solve the complex problem of mapping neural networks onto hierarchical neuromorphic hardware. By utilizing a **Generative Stochastic Block Model (SBM)** and **Simulated Annealing (MCMC)**, NeToChi finds hardware-compliant mappings that maximize connection likelihood while respecting physical constraints.

---

## ✨ Features

- **Generative SBM Approach**: Moves beyond rigid heuristics to a statistical model of hardware compliance.
- **Hierarchical Hardware Modeling**: Support for multi-level router hierarchies, core partitions, and slice-based fan-in constraints.
- **Optimized MCMC Inference**: Simulated annealing with aggressive likelihood optimization for fast convergence.
- **Extensible Architecture**: Modular design for adding new mapping objectives, network generators, or hardware configurations.
- **Hybrid Mapping**: Combines hSBM-based community detection with PCA-driven placement for rapid initialization.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/username/netochi.git
cd netochi

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install .
```

> [!NOTE]
> NeToChi requires `graph-tool`, which is best installed via `conda` or system package managers for performance.

---

## 🚀 Quick Start

```python
import graph_tool.all as gt
from netochi.mapping.hardware_config import HardwareConfig
from netochi.pipeline.objectives import MappingProblem, LikelihoodObjective
from netochi.mapping.mcmc_mapper import Section5SBM_MCMCMapper

# 1. Define your hardware
hw_config = HardwareConfig(
    nodes_per_router=2,
    neurons_per_core=15,
    router_levels=2
)

# 2. Load or generate your graph
# g = gt.load_graph("my_network.xml.gz")
g = gt.price_network(100) # Example graph

# 3. Setup the optimization problem
problem = MappingProblem(graph=g, objective=LikelihoodObjective(), initial_hardware=hw_config)

# 4. Run the MCMC Mapper
mapper = Section5SBM_MCMCMapper(problem)
optimized_cost = mapper.optimize(iterations=5000)

print(f"Optimization complete. Final Cost: {optimized_cost}")
```

---

## 📚 Documentation

Detailed documentation and implementation guides can be found in the [docs/](docs/) folder:
- [SBM & MCMC Implementation Guide](docs/SBM_MCMC_Guide.md)
- [QAP Mapping Mathematics](docs/QAP_Mapping_Mathematics.md)

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
