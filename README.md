# NeToChi

**Mapping Neural Networks onto Neuromorphic Hardware with Hierarchical Routing**

NeToChi (*Neuromorphic To Chip*) is a Python research code base developed in the student project *Small-World Network Mapping* by Laura Erb and Laurenz Wurzel. It addresses the problem of assigning the neurons of a given network to a neuromorphic chip whose cores are connected by a hierarchical routing tree and whose connectivity is therefore restricted. The repository contains a formalization of these hardware constraints, a set of mapping algorithms, synthetic network generators, and an experiment pipeline for comparing the algorithms.

---

## Problem Setting

The hardware model follows the project report ([docs/Semester_Project__Small_World_Network_Mapping.pdf](docs/Semester_Project__Small_World_Network_Mapping.pdf)).

**Hardware.** Let $R$ denote the number of children per router and $d_{max}$ the depth of the routing tree. The chip consists of $C = R^{d_{max}}$ cores, each holding $n$ neurons. The *distance* $d$ between two cores is the level of their lowest common router ($d = 0$ within a core).

**Connectivity constraints.** A connection on the chip must be admitted both by the *fan-in* and by the *fan-out* configuration; the set of realizable connections is $E_{HW} = F_I \cap F_O$.

- *Fan-in (source-based routing):* for every distance $d$, a neuron listens to exactly one of $S_d$ predefined slices of local addresses. The same slice is used for all source cores at that distance.
- *Fan-out (destination-based routing):* every source neuron stores a hierarchical bit string with one routing mask per level, which determines the set of target cores its spikes are delivered to.

**Mapping task.** Given a directed graph $G = (V, E)$ and a hardware configuration, the task is to assign each neuron $i$ a core $c(i)$, a local address $x(i)$ within that core, and a slice $s_{i,d}$ for every distance $d$, such that no two neurons occupy the same address. Mappings are evaluated by the number (or fraction) of *inconsistent* edges, i.e. edges of $G$ that cannot be realized on the chip under the chosen configuration.

The task decomposes into three phases, which several of the implemented algorithms treat separately:

1. **Core assignment** (hierarchical clustering of the neurons into cores),
2. **Local address assignment** (ordering of the neurons within each core),
3. **Slice assignment** (selection of the listening slice per neuron and distance).

Given the core and local address assignments, the optimal slice assignment can be computed exactly in $O(|E| \cdot 2^{d_{max}})$. The first two phases are NP-hard in general; the computational complexity of the joint problem remains open.

---

## Implemented Methods

All mappers operate on a `MosaicMappingInput` (network and hardware configuration) and return a mapping state that can be evaluated with the objectives in `netochi/objectives/`.

| Method | Description | Module |
|---|---|---|
| Random | Uniformly random core, address, and slice assignment; used as a baseline. | `mapping/random_mapper.py` |
| Greedy | Degree-ordered greedy placement of neurons next to already placed neighbors. | `mapping/greedy_mapper.py` |
| Simulated annealing | Joint optimization of core and address assignment; the optimal slice assignment is computed in every iteration. A variant is initialized with a clustering. | `mapping/simulated_annealing_mapper.py`, `mapping/simulated_annealing_mapper_cluster_init.py` |
| Three-step mapper | Solves the three phases sequentially; clusterers and address assigners can be combined freely. | `mapping/three_step_mapping/` |
| Feedback-loop mapper | Alternates exact slice steps, linear-assignment index steps, and periodic core moves guided by reweighted violated edges. | `mapping/feedback_loop_mapper.py` |
| MILP mapper | Mixed-integer linear programming formulation; applicable to small instances. | `mapping/ilp_mapper.py` |
| SBM / MCMC mapper | Generative stochastic block model of hardware-compliant connectivity, optimized by simulated annealing in `graph-tool`. | `mapping/mcmc/` |
| SDP recovery mapper | Recovery of a planted hierarchical structure via semidefinite programming (binary routers only). | `mapping/theoretical_bounds_mapper.py` |

The three-step mapper provides the following components:

- **Clustering:** hierarchical community detection (HCD) [2], recursive partitioning with KaHyPar [3] (with and without hyperedges) and KaMinPar [4], a quadratic assignment (QAP) formulation solved with the FAQ heuristic, the hierarchical stochastic block model (hSBM) [5], and recursive spectral bisection.
- **Local address assignment:** LP relaxation with pipage rounding, SDP relaxation with hyperplane rounding, simulated annealing, Kernighan–Lin bisection, a PCA-based ordering, and recursive bisection on the co-fan-in similarity.
- **Slice assignment:** exact optimal assignment.

---

## Network Generators

The generators in `netochi/input_generator/` produce synthetic inputs:

- **Mosaic:** graphs that are feasible for a given hardware by construction. Slices are drawn at random and each hardware-admissible edge is sampled with probability $p$. The generating assignment serves as a ground-truth reference.
- **Noisy Mosaic:** as above, with additional edges outside the admissible set sampled with probability $\epsilon$.
- **Erdős–Rényi:** directed random graphs with edge probability $p$.
- **sWTA / WTA:** soft winner-take-all architectures with excitatory clusters and a global inhibitory population, following [6].

---

## Installation

The code requires Python 3.12 or 3.13 and [`graph-tool`](https://graph-tool.skewed.de/), which is not available on PyPI and is most easily installed via conda-forge.

**With pixi (recommended).** The environment is defined in `pyproject.toml` and pinned in `pixi.lock`; it installs graph-tool from conda-forge, the remaining packages from PyPI, and NeToChi itself in editable mode.

```bash
git clone https://github.com/laurenzlrz/NeToChi.git
cd NeToChi
pixi install
pixi run test          # python -m pytest tests
pixi run experiments   # python -m netochi.run_experiments
pixi shell             # activate the environment in the current shell
```

**With conda and pip.**

```bash
conda env create -f environment.yml
conda activate netochi
pip install -r requirements.txt
```

---

## Usage

### Mapping a single network

```python
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
from netochi.mapping.feedback_loop_mapper import FeedbackLoopConfig
from netochi.objectives.obj_inconsistency import InconsistencyRelativeObjectiveConfig

# Binary routing tree with 2 levels (4 cores) and 16 neurons per core
hw = MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=16, router_levels=2, slice_factor=2)

# Feasible network for this hardware, edge probability 0.5
mapping_input = MosaicNetworkConfig(hw_config=hw, probability=0.5, seed=1).create().generate()

state = FeedbackLoopConfig().create().run(mapping_input)

result = InconsistencyRelativeObjectiveConfig().create().evaluate(state)
print(f"Fraction of inconsistent edges: {result.value:.3f}")
```

### Running experiments

The experiments reported in the project are configured in [`netochi/run_experiments.py`](netochi/run_experiments.py). The hardware configurations, network generators, mappers, objectives, and random seeds are set in the configuration block at the top of the file. The script is executed with

```bash
python -m netochi.run_experiments
```

The pipeline writes summaries, CSV exports, and plots of the selected objectives and runtimes to the configured output directory.

### Tests

```bash
pytest tests
```

---

## Repository Structure

```
netochi/
├── definitions/        constants, exceptions, and shared base classes
├── input_generator/    hardware configuration and network generators
├── mapping/            mapping algorithms (see "Implemented Methods")
├── objectives/         evaluation metrics (inconsistencies, hardware size, likelihood, clustering scores)
├── pipeline/           experiment runner, validation, and result consumers
├── result_processing/  aggregation and plotting of experiment results
└── visualization/      plots of adjacency matrices, clusterings, and routing hierarchies
docs/                   project report and notes on individual methods
tests/                  unit and integration tests
```

Additional notes on individual methods:

- [Generative SBM and MCMC inference](docs/SBM_MCMC_Guide.md)
- [Joint hardware and mapping inference via minimum description length](docs/joint_hardware_inference.md)
- [QAP formulation of the mapping problem](docs/QAP_Mapping_Mathematics.md)
- [Greedy mapper](docs/greedy_mapper_algorithm.md)

---

## References (WIP)

1. L. Erb and L. Wurzel. *Student Project: Small-World Network Mapping.* Project report, 2026. [docs/Semester_Project__Small_World_Network_Mapping.pdf](docs/Semester_Project__Small_World_Network_Mapping.pdf)
2. M. T. Schaub, J. Li, and L. Peel. Hierarchical community structure in networks. *Physical Review E*, 107(5):054305, 2023. doi:[10.1103/PhysRevE.107.054305](https://doi.org/10.1103/PhysRevE.107.054305)
3. S. Schlag, V. Henne, T. Heuer, H. Meyerhenke, P. Sanders, and C. Schulz. k-way hypergraph partitioning via n-level recursive bisection. In *Proceedings of the 18th Workshop on Algorithm Engineering and Experiments (ALENEX)*, pp. 53–67, 2016. doi:[10.1137/1.9781611974317.5](https://doi.org/10.1137/1.9781611974317.5)
4. L. Gottesbüren, T. Heuer, P. Sanders, C. Schulz, and D. Seemaier. Deep multilevel graph partitioning. In *29th Annual European Symposium on Algorithms (ESA)*, pp. 48:1–48:17, 2021. doi:[10.4230/LIPIcs.ESA.2021.48](https://doi.org/10.4230/LIPIcs.ESA.2021.48)
5. T. P. Peixoto. Hierarchical block structures and high-resolution model selection in large networks. *Physical Review X*, 4(1):011047, 2014. doi:[10.1103/PhysRevX.4.011047](https://doi.org/10.1103/PhysRevX.4.011047)
6. D. Zendrikov, S. Solinas, and G. Indiveri. Brain-inspired methods for achieving robust computation in heterogeneous mixed-signal neuromorphic processing systems. *Neuromorphic Computing and Engineering*, 3(3):034002, 2023. doi:[10.1088/2634-4386/ace64c](https://doi.org/10.1088/2634-4386/ace64c)

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
