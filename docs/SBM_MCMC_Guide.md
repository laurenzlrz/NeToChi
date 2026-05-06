# Section 5: Generative SBM & MCMC Implementation Guide

This document explains the Python implementation of the "Non-Hierarchic Extended Stochastic Block Model" (from Section 5 of the PDF) and how we use Markov Chain Monte Carlo (MCMC) to infer the optimal neuromorphic hardware mapping.

## 1. The Stochastic Framework

The goal of Section 5 is to map a directed graph $G = (V,E)$ onto a neuromorphic hardware structure defined by cores, addresses, and slices. Instead of a rigid two-phase heuristic, we model this as a single **Generative Statistical Model**. 

We assume the network was generated from a parameterized Poisson process where the expected number of edges between node $j$ and node $i$ is proportional to:
1. Their expected degrees ($\gamma_i^{in}$ and $\gamma_j^{out}$).
2. A hardware-compliant coupling matrix $W_{ij}$.

### The Bayesian Foundation

The entire approach is rooted in Bayes' theorem. We want to find the mapping parameters $\Theta = \{c, x, s\}$ that best explain the observed graph $A$:

$$P(\Theta \mid A) = \frac{P(A \mid \Theta) \cdot P(\Theta)}{P(A)}$$

Where:
*   $P(A \mid \Theta)$ is the **likelihood**: "Given a specific hardware mapping, how probable is the observed network?" This is what the generative SBM model computes.
*   $P(\Theta)$ is the **prior**: We assume a uniform prior over all valid mappings, so it becomes a constant and drops out.
*   $P(A)$ is the **evidence**: This is a normalization constant over all possible mappings. It is intractable to compute, but since we only ever compare *ratios* of posteriors (in the Metropolis-Hastings step), it cancels out.

Because the prior is uniform and the evidence cancels, maximizing the posterior $P(\Theta \mid A)$ reduces to maximizing the likelihood $P(A \mid \Theta)$, which is equivalent to maximizing the log-likelihood $\mathcal{L}(\Theta)$. This is the **Maximum Likelihood Estimation (MLE)** simplification that makes the entire MCMC tractable.

### The Coupling Matrix $W_{ij}$
The variables we are trying to optimize are the mapping assignments:
*   $c_i$: The core assigned to node $i$.
*   $x_i$: The local address of node $i$ inside the core.
*   $s_{i, d}$: The specific slice that node $i$ listens to at distance $d$.

Based on these assignments, the indicator function $\Delta_{ij} \in \{0, 1\}$ checks if a connection is hardware-compliant (Fan-In constraints). The coupling matrix is defined stochastically:
$$W_{ij} = \alpha \Delta_{ij} + \epsilon (1 - \Delta_{ij})$$
where $\alpha$ is a high probability for compliant edges and $\epsilon$ is a noise probability for non-compliant edges.

### From Observed Degrees to Expected Degrees

The generative model uses **expected degrees** $\gamma_i^{in}$ and $\gamma_j^{out}$ as parameters that control how likely each node is to form connections. In a theoretical derivation, these would be latent variables that we would need to estimate jointly with the mapping.

However, in our implementation (`likelihood_state.py`, lines 24-26), we make a deliberate simplification:

```python
self.k_in = graph.get_in_degrees(graph.get_vertices())   # observed in-degrees
self.k_out = graph.get_out_degrees(graph.get_vertices())  # observed out-degrees
```

We substitute the **observed** degrees $k_i^{in}$, $k_j^{out}$ directly in place of the expected degrees $\gamma_i^{in}$, $\gamma_j^{out}$. This is justified by a key statistical argument:

1.  **Law of Large Numbers**: For sufficiently large and dense networks, the observed degree of a node converges to its expected degree: $k_i \xrightarrow{N \to \infty} \gamma_i$. Since our benchmark networks typically have hundreds to thousands of neurons with many connections each, this approximation is accurate.
2.  **Computational Simplification**: Estimating $\gamma$ as a separate latent variable would require an additional inference loop (e.g., EM algorithm), adding significant complexity with marginal accuracy gains for our network sizes.
3.  **Standard Practice**: This substitution is standard in the degree-corrected SBM literature (Karrer & Newman, 2011). It is the same approximation used in `graph-tool`'s own SBM inference.

**In summary**: The observed degrees $k$ are a "plug-in estimator" for the true expected degrees $\gamma$. This decouples degree estimation from mapping inference, allowing us to focus the entire MCMC exclusively on optimizing the hardware assignments $\Theta = \{c, x, s\}$.

### The Likelihood Function $\mathcal{L}(\Theta)$
The objective is to find the mapping that maximizes the log-likelihood of observing the actual graph adjacency matrix $A$:
$$\mathcal{L}(\Theta) \propto -m \log Z + \sum_{i,j} A_{ij} \log W_{ij}$$
Here, $Z$ is the intrinsic normalization constant: $Z = \sum_{u,v} \gamma_u^{in} \gamma_v^{out} W_{uv}$.

#### Computational Optimization for $Z$
A naive calculation of $Z$ requires iterating over all pairs of nodes $O(N^2)$, which is computationally impossible to do millions of times during inference. 
In `likelihood_state.py`, we aggressively optimize this. We recognize that $\Delta_{ij}$ only depends on the target's slice and the source's local address. We pre-calculate a 3D tensor `slice_out_mass[core, distance, slice]` that tracks the sum of all out-degrees ($\gamma^{out}$) in that specific partition. This allows us to recompute the entire complex $Z$ partition function in $O(N \cdot d_{max})$ time.

---

## 2. How We Use MCMC (Simulated Annealing)

The likelihood function is heavily discrete, non-differentiable, and NP-hard to maximize directly. We cannot use standard gradient descent. Instead, we use **Markov Chain Monte Carlo (MCMC)**, specifically **Simulated Annealing**, implemented in `mcmc_mapper.py`.

### The MCMC Loop
The algorithm starts with a completely random hardware assignment (generated by `RandomMapper`). Over `iterations` steps, we attempt to "mutate" the mapping to find better states.

#### 1. Proposing a Move
At each step, we randomly select a node and propose one of two moves:
*   **Node Swap:** We pick a second random node and swap their core ($c$) and local address ($x$) assignments.
*   **Slice Mutation:** We randomly pick a hierarchical distance $d$, and change the slice $s_{i,d}$ that the node listens to.

#### 2. Evaluating the Proposal
After proposing the move, we recalculate the log-likelihood $\mathcal{L}_{new}$. 

#### 3. Metropolis-Hastings Acceptance
We compare the new likelihood to the old likelihood ($\mathcal{L}_{old}$):
*   **If $\mathcal{L}_{new} > \mathcal{L}_{old}$**: We immediately **accept** the move. The mapping improved!
*   **If $\mathcal{L}_{new} \le \mathcal{L}_{old}$**: We might still accept a "bad" move with probability $p = e^{(\mathcal{L}_{new} - \mathcal{L}_{old}) / T}$. This is the crucial stochastic element. It allows the algorithm to occasionally step "downhill" to escape local minima traps in the complex mapping landscape.

Note how Bayes' theorem enables this: since the prior $P(\Theta)$ is uniform and $P(A)$ cancels in the ratio, the acceptance ratio simplifies to a pure likelihood ratio. We never need to compute the intractable evidence $P(A)$.

#### 4. The Cooling Schedule (Temperature $T$)
The parameter $T$ is the "Temperature".
*   We start at a high temperature, meaning the MCMC chain is highly erratic and explores a wide variety of mappings (accepting many bad moves).
*   As iterations progress, we apply an exponential decay to $T$. The chain "cools down" and gradually becomes greedier, fine-tuning the mapping until it settles into a maximum likelihood peak.

---

## 3. The `GraphToolCompatibleState` Adapter

The `graph-tool` library provides a highly optimized `mcmc_anneal` function that implements the cooling schedule, sweep logic, and convergence tracking. However, it expects a state object with a specific interface. Rather than reimplementing simulated annealing from scratch, we use the **Adapter Pattern** to make our `MappingState` compatible.

### The Interface Contract

`graph-tool`'s `mcmc_anneal` requires a state object that provides two methods:

| Method | Expected Signature | Purpose |
|---|---|---|
| `entropy(**kwargs)` | `() -> float` | Returns the current "energy" of the system. `mcmc_anneal` **minimizes** this value. |
| `mcmc_sweep(beta, **kwargs)` | `(beta: float) -> (dS, nattempts, nmoves)` | Performs one full sweep of $N$ move-attempts at inverse temperature $\beta = 1/T$, returning the change in entropy, number of attempts, and number of accepted moves. |

### How the Adapter Works

The `GraphToolCompatibleState` class (`mcmc_mapper.py`) wraps our `MappingState` and translates between the two worlds:

#### `entropy()` → Our Cost Function
```python
def entropy(self, **kwargs) -> float:
    if self.objective:
        return self.objective.evaluate(self.state)
    return -self.state.log_likelihood()
```
Since `mcmc_anneal` **minimizes** entropy, we return the **negative** log-likelihood (or a custom objective's cost). Minimizing $-\mathcal{L}$ is equivalent to maximizing $\mathcal{L}$.

#### `mcmc_sweep()` → Our Move Proposals + Metropolis-Hastings
This method performs one "sweep" of $N$ individual move attempts. For each attempt:

1.  **Choose move type** (50/50): Node swap or slice mutation.
2.  **Apply the move** tentatively to the internal `MappingState`.
3.  **Evaluate** `entropy()` to get the new energy.
4.  **Accept or reject** using the Metropolis criterion with the $\beta$ (inverse temperature) provided by `mcmc_anneal`:
    *   Accept if $\Delta E < 0$ (improvement), or with probability $e^{-\Delta E \cdot \beta}$ (stochastic uphill).
    *   If rejected, **revert** the state arrays to their previous values.
5.  **Track the best state** seen so far (lines 30-35), so that even if the chain wanders away from the optimum later, we can restore it.

#### Best-State Tracking
A critical detail: the MCMC chain does not necessarily end at its best state. The chain might accept a few bad moves near the end, or it might be killed by the 10-second timeout mid-sweep. The adapter tracks the lowest energy seen:

```python
def _save_best(self, energy: float) -> None:
    if energy < self._best_energy:
        self._best_energy = energy
        self._best_c = self.state.c.copy()
        self._best_x = self.state.x.copy()
        self._best_s = self.state.s.copy()
```

After `mcmc_anneal` completes (or times out), `restore_best()` is called to roll back to the best mapping discovered during the entire run.

### Baseline Comparison
To prove the MCMC is effectively solving the statistical model, we compare its final optimized likelihood against a `RandomMapper` baseline (which represents the statistical energy of a completely random mapping).
