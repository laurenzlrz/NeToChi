# Quadratic Assignment Problem (QAP) for Neuromorphic Mapping

The mapping of a neural network graph onto a modular neuromorphic hardware architecture can be framed as a **Quadratic Assignment Problem (QAP)**. This document outlines the mathematical transformation used in the `QAPMapper`.

## 1. Problem Definition
We are given:
- A directed graph $G = (V, E)$ with adjacency matrix $A$, where $A_{ij} = 1$ if there is an edge from neuron $i$ to neuron $j$.
- A hardware topology with $M$ slots. Each slot $k$ is defined by a pair $(c_k, x_k)$, where $c_k$ is the core ID and $x_k$ is the local address within that core.
- Hardware constraints: An edge $i \to j$ mapped to slots $(c_i, x_i)$ and $(c_j, x_j)$ is valid if:
    1. $c_i = c_j$ (Distance $d=0$), OR
    2. $d = \text{dist}(c_j, c_i) > 0$ AND $x_i$ falls within the listening slice assigned to target $j$ at distance $d$.

## 2. Transformation to QAP
The standard QAP seeks to find a permutation $\pi$ of $N$ items that minimizes (or maximizes) the objective:
$$ \max_{\pi} \sum_{i,j} A_{ij} W_{\pi(i)\pi(j)} $$
where $A$ is the flow matrix and $W$ is the affinity matrix.

### 2.1 The Affinity Matrix $W$
In our transformation, $W_{kl}$ represents the **expected validity** of a connection from hardware slot $k$ to hardware slot $l$. 

Since the target neuron at slot $l$ can choose exactly one slice of local addresses to listen to at any distance $d > 0$, and there are $S_d$ available slices at that distance, the probability that a connection from slot $k$ is valid (without knowing the specific slice selection) is:
$$ W_{kl} = \begin{cases} 1.0 & \text{if } \text{dist}(c_l, c_k) = 0 \\ \frac{1}{S_d} & \text{if } \text{dist}(c_l, c_k) = d > 0 \\ 0 & \text{if unreachable} \end{cases} $$

Where $S_d$ is the `num_slices_at_distance(d)`.

### 2.2 Solving the Problem
We use the **Fast Approximate QAP (FAQ)** algorithm (gradient-based relaxation) provided by `scipy.optimize.quadratic_assignment`. 
1. **Input A**: The adjacency matrix of the neural network (padded with zeros if $N < M$).
2. **Input B**: The affinity matrix $W$ calculated for all pairs of hardware slots.
3. **Maximization**: We solve for the permutation that maximizes the trace of $A P W^T P^T$.

## 3. Post-Processing: Greedy Slice Selection
The QAP solution provides the optimal core ($c$) and local address ($x$) for each neuron. However, it does not explicitly assign the listening slices ($s$). 

After the QAP permutation is obtained, the `QAPMapper` performs a deterministic greedy pass:
For each target neuron $j$ and each distance $d > 0$:
$$ s_{j,d} = \arg \max_{\sigma \in \{0, \dots, S_d-1\}} \sum_{i \in \text{InEdges}(j)} \mathbb{I}(x_i \in \text{Slice}(d, \sigma)) $$
This ensures that the final mapping strictly adheres to hardware constraints while maximizing the actual number of valid connections based on the fixed $(c, x)$ placement.
