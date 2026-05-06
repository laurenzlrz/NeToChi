# Greedy Mapper Algorithm

The Greedy Mapper is a heuristic approach to mapping artificial neural networks onto hierarchical neuromorphic hardware (like Mosaic). It prioritizes mapping connected neurons close to each other while respecting hard capacity constraints.

## Algorithm Phases

### 1. Pre-Processing: Node Ordering
The algorithm begins by calculating the total degree (sum of in-degree and out-degree) for every neuron in the input graph. 
- **Sorting**: Neurons are sorted in descending order of their total degree.
- **Rationale**: Highly connected "hub" neurons are mapped first to ensure they have the best chance of being placed in optimal locations before hardware resources become scarce.

### 2. Phase I: Core Assignment
The algorithm iterates through the sorted list of neurons and assigns each to a physical core.

1.  **Core Scoring**: For each candidate core, a "proximity score" is calculated. The score is the number of neighbors of the current neuron that have already been mapped to that specific core.
2.  **Greedy Selection**: The neuron is assigned to the core with the highest proximity score that still has free capacity (`neurons_per_core`).
3.  **Local Address Assignment**: Within the chosen core, the neuron is assigned the next available local address ($x \in [0, \text{neurons\_per\_core}-1]$).

### 3. Phase II: Slice Selection (Fan-In Optimization)
Once all neurons are assigned to cores and local addresses, the algorithm optimizes the "listening slices" ($s$) for each neuron at every hierarchy level.

1.  **Per-Level Optimization**: For each neuron and each hierarchy distance $d \in [1, \text{max\_distance}]$:
    - The algorithm evaluates every possible slice index $s$ available at that distance.
    - It counts how many of the neuron's incoming neighbors ($src$) are currently valid if this slice $s$ is chosen.
    - **Validity Criterion**: A connection is valid if the source core is at distance $d$ and the source's local address falls within the bounds of slice $s$.
2.  **Selection**: The slice index that maximizes the number of valid incoming connections is selected for that neuron and level.

## Complexity
- **Time Complexity**: $O(N \cdot K \cdot C + N \cdot D \cdot S \cdot \text{avg\_degree})$, where $N$ is nodes, $K$ is average degree, $C$ is cores, $D$ is levels, and $S$ is max slices.