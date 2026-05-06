# Joint Hardware-Mapping Inference via Minimum Description Length

This document extends the Section 5 SBM framework to jointly infer the **optimal hardware size** alongside the mapping parameters. Instead of taking the hardware configuration $H$ as fixed, we treat it as a latent variable and use the **Minimum Description Length (MDL)** principle to find the smallest hardware that still explains the network well.

---

## 1. Motivation

In Section 5, we solve: given a fixed hardware $H$, find the best mapping $\Theta^*(H)$. But this raises a meta-question: **how do we choose $H$ in the first place?**

- Too few cores → neurons are overcrowded, Fan-In constraints are violated, likelihood is poor.
- Too many cores → perfect fit, but wasteful: we allocated silicon we didn't need.

We want the **Goldilocks hardware**: the smallest configuration that still accommodates the network's connectivity well. This is a classic **model selection** problem, and the MDL principle provides the mathematically principled answer.

### Connection to Peixoto's Work

This extension mirrors exactly how `graph-tool` handles model selection for SBMs:

| Peixoto (Community Detection) | Our Extension (Hardware Mapping) |
|---|---|
| Partition $b$ (block assignments) | Mapping $\Theta = \{c, x, s\}$ |
| Number of blocks $B$ | Hardware config $H = \{K, N_c, \ldots\}$ |
| $\text{DL}(A, b, B)$ minimized jointly | $\Sigma(A, \Theta, H)$ minimized jointly |

Peixoto minimizes the description length to find the right number of communities. We minimize it to find the right number of cores.

---

## 2. Mathematical Framework

### 2.1 The Full Posterior

We seek the joint MAP estimate over mapping $\Theta$ and hardware $H$:

$$(\Theta^*, H^*) = \arg\max_{\Theta, H} \; P(\Theta, H \mid A)$$

By Bayes' theorem:

$$P(\Theta, H \mid A) \propto P(A \mid \Theta, H) \cdot P(\Theta \mid H) \cdot P(H)$$

Taking the negative logarithm converts maximization to minimization of the **total description length**:

$$\boxed{\Sigma(\Theta, H) = \underbrace{-\mathcal{L}(\Theta, H)}_{\text{Data cost}} + \underbrace{\mathcal{C}(\Theta \mid H)}_{\text{Mapping cost}} + \underbrace{\mathcal{C}(H)}_{\text{Hardware cost}}}$$

Each term has a clear information-theoretic interpretation:
1. **Data cost** $-\mathcal{L}$: How many nats to describe the *residual* — the part of the graph not explained by the model.
2. **Mapping cost** $\mathcal{C}(\Theta \mid H)$: How many nats to describe *which* mapping we chose, given the hardware.
3. **Hardware cost** $\mathcal{C}(H)$: How many nats to describe the hardware configuration itself.

### 2.2 Term 1: Data Cost (Negative Log-Likelihood)

This is identical to the Section 5 likelihood. Given hardware $H$ with $K$ active cores and mapping $\Theta$:

$$-\mathcal{L}(\Theta, H) = m \log Z(\Theta, H) - \sum_{(i,j) \in E} \log W_{ij}(\Theta, H)$$

where:
- $W_{ij} = \alpha \cdot \Delta_{ij} + \epsilon \cdot (1 - \Delta_{ij})$ is the coupling matrix.
- $Z = \sum_{u,v} \gamma_u^{in} \gamma_v^{out} W_{uv}$ is the partition function.
- $\Delta_{ij} \in \{0,1\}$ indicates Fan-In compliance.

**Crucially**, both $W$ and $Z$ depend on $H$ through the core distances and slice structure. Changing $K$ changes which cores exist, which changes all pairwise distances, which changes $\Delta$, $W$, and $Z$.

### 2.3 Term 2: Mapping Description Length $\mathcal{C}(\Theta \mid H)$

Given hardware $H = \{K, N_c, D_{\max}, \{S_d\}_{d=1}^{D_{\max}}\}$, the mapping $\Theta = \{c, x, s\}$ assigns each of $N$ neurons a core, a local address, and slice selections. We need to encode this assignment.

**Core assignment** $c$: Each neuron is assigned to one of $K$ cores. Under a uniform prior:
$$\mathcal{C}(c) = N \log K$$

**Local address assignment** $x$: Given the core assignment, each neuron in core $k$ receives a unique address from $\{0, \ldots, N_c - 1\}$. If core $k$ contains $n_k$ neurons, the number of valid address permutations is $\frac{N_c!}{(N_c - n_k)!}$, giving:
$$\mathcal{C}(x \mid c) = \sum_{k=0}^{K-1} \log \binom{N_c}{n_k} + \log(n_k!)$$

For dense cores ($n_k \approx N_c$), this simplifies via Stirling's approximation. For sparse cores ($n_k \ll N_c$), we get $\mathcal{C}(x \mid c) \approx N \log N_c$.

**Slice assignment** $s$: Each neuron independently selects a listening slice at each distance $d \in \{1, \ldots, D_{\max}\}$:
$$\mathcal{C}(s) = N \sum_{d=1}^{D_{\max}} \log S_d$$
where $S_d = \min(\text{slice\_factor}^d, N_c)$ is the number of slices at distance $d$.

**Total mapping cost**:
$$\mathcal{C}(\Theta \mid H) = N \log K + \sum_{k=0}^{K-1} \left[\log \binom{N_c}{n_k} + \log(n_k!)\right] + N \sum_{d=1}^{D_{\max}} \log S_d$$

**Simplified form** (used in implementation, valid when $n_k \ll N_c$):
$$\mathcal{C}(\Theta \mid H) \approx N \left[\log K + \log N_c + \sum_{d=1}^{D_{\max}} \log S_d \right]$$

Note: Since $N_c$, $D_{\max}$, and $\{S_d\}$ are fixed by the physical chip, the only term that varies with hardware selection is $N \log K$. This is the key penalty: **each additional core costs $N \log(K/(K-1))$ nats**.

### 2.4 Term 3: Hardware Description Length $\mathcal{C}(H)$

We encode the hardware configuration using Rissanen's universal prior for integers. For a positive integer $n$, the universal code length is:

$$\log^*(n) = \log_2(c_0) + \log_2(n) + \log_2(\log_2(n)) + \log_2(\log_2(\log_2(n))) + \cdots$$

where only positive terms are summed, and $c_0 \approx 2.865$ is a normalizing constant.

For the Mosaic hardware, the primary variable is the number of active cores $K$:

$$\mathcal{C}(H) = \log^*(K)$$

Since $N_c$, the slice factor, and the router topology are fixed by the physical chip, they contribute a constant to $\mathcal{C}(H)$ and can be dropped from the optimization.

### 2.5 The Trade-Off

The total energy creates a natural competition:

$$\Sigma(\Theta, H) = \underbrace{-\mathcal{L}(\Theta, H)}_{\searrow \text{ as } K \nearrow} + \underbrace{N \log K + \log^*(K)}_{\nearrow \text{ as } K \nearrow}$$

- **More cores**: The data cost decreases (better routing options, fewer Fan-In violations), but the description length increases.
- **Fewer cores**: The description length is small, but neurons are crowded and many edges become invalid.

The optimal $K^*$ is the point where adding one more core no longer improves the likelihood enough to justify the extra description length.

### 2.6 Bounds on $K$

The search space for $K$ is bounded:

$$K_{\min} = \left\lceil \frac{N}{N_c} \right\rceil \leq K \leq K_{\max} = N_r^{L}$$

where $K_{\min}$ is the minimum number of cores needed to physically fit all neurons, and $K_{\max} = \text{nodes\_per\_router}^{\text{router\_levels}}$ is the total number of physical cores on the chip.

---

## 3. The Joint MCMC Algorithm

### 3.1 Extended Move Set

We extend the Section 5 MCMC with **hardware moves** that change $K$:

| Move Type | Changes | Probability |
|---|---|---|
| Node Swap | $\Theta$ (core + address of two nodes) | $p_{\text{swap}}$ |
| Slice Mutation | $\Theta$ (slice at one distance) | $p_{\text{slice}}$ |
| Core Addition | $H$ ($K \to K+1$) | $p_{\text{add}}$ |
| Core Removal | $H$ ($K \to K-1$) | $p_{\text{remove}}$ |

### 3.2 Core Addition Move ($K \to K+1$)

**Precondition**: $K < K_{\max}$.

**Procedure**:
1. Select a random occupied core $k$ with $n_k > 1$ neurons.
2. Generate a split ratio: each neuron in core $k$ is moved to the new core $K$ with probability $0.5$ (auxiliary Bernoulli variables $u_i$).
3. Neurons that move to core $K$ receive fresh random local addresses in $\{0, \ldots, N_c - 1\}$.
4. Slice assignments for moved neurons are re-randomized (the routing distances have changed).

**Energy change**:
$$\Delta\Sigma = \Delta(-\mathcal{L}) + N \log\frac{K+1}{K} + \log^*(K+1) - \log^*(K)$$

The first term requires recomputing the likelihood (since many $\Delta_{ij}$ values change). The second and third terms are the DL penalty for the extra core.

### 3.3 Core Removal Move ($K \to K-1$)

**Precondition**: $K > K_{\min}$.

**Procedure**:
1. Select a random core $k$ (prefer cores with few neurons, or select uniformly).
2. Redistribute all $n_k$ neurons from core $k$ to other cores:
   - For each neuron, assign it to a random core $k' \neq k$ that has capacity ($n_{k'} < N_c$).
   - Assign a random available local address in core $k'$.
3. Re-randomize slice assignments for relocated neurons.
4. Renumber all cores $> k$ by decrementing their index by 1.

**Energy change**: Same formula as addition, but with $K-1$ replacing $K+1$.

### 3.4 Acceptance Criterion

All moves (mapping and hardware) use the Metropolis-Hastings criterion with the full energy $\Sigma$:

$$P(\text{accept}) = \min\left(1, \; \exp\left(-\beta \cdot \Delta\Sigma\right) \cdot \frac{q(\text{reverse})}{q(\text{forward})}\right)$$

where $\beta$ is the inverse temperature (from simulated annealing) and $q$ is the proposal probability.

**Proposal ratio for split/merge**:
- Forward (split core $k$ into $k$ and $K$): $q_{\text{fwd}} = \frac{1}{K} \cdot 2^{-n_k}$ (choose core $k$, then flip coins for each neuron).
- Reverse (merge cores $k$ and $K$): $q_{\text{rev}} = \frac{1}{\binom{K+1}{2}}$ (choose which two cores to merge).

The ratio $q_{\text{rev}}/q_{\text{fwd}}$ ensures detailed balance is maintained across dimension-changing moves (Reversible Jump MCMC, Green 1995).

### 3.5 Cooling Schedule

The simulated annealing schedule is the same as Section 5:
- Start with high temperature (low $\beta$): explore widely, including large hardware changes.
- Cool exponentially: gradually focus on fine-tuning.
- At low temperature: only accept improving moves.

---

## 4. Computational Considerations

### 4.1 Efficient $Z$ Recomputation

When $K$ changes, the partition function $Z$ must be recomputed from scratch because the core distance matrix changes globally. This is $O(N \cdot D_{\max})$ per evaluation. For hardware moves (which are rare compared to mapping moves), this is acceptable.

### 4.2 Lazy Evaluation

We can avoid full likelihood recomputation for hardware moves by:
1. Computing $\Delta(-\mathcal{L})$ incrementally for the relocated neurons only.
2. Caching the slice-out-mass tensor and updating only affected entries.

### 4.3 Move Frequency

Hardware moves should be rare compared to mapping moves. A typical schedule:
- $p_{\text{swap}} = 0.45$, $p_{\text{slice}} = 0.45$, $p_{\text{add}} = 0.05$, $p_{\text{remove}} = 0.05$.

This ensures the mapping has time to equilibrate between hardware changes.

---

## 5. Interpretation

The optimal $(K^*, \Theta^*)$ returned by the algorithm answers two questions simultaneously:

1. **How large should the hardware be?** $K^*$ is the minimum number of cores needed to achieve a good mapping, as determined by the MDL principle.
2. **How should neurons be mapped?** $\Theta^*$ is the best assignment for that hardware size.

This is directly useful for:
- **Chip sizing**: Given a target network, determine the smallest chip that can run it.
- **Resource allocation**: On a multi-tenant chip, determine how many cores to allocate to each workload.
- **Architecture exploration**: Compare different $(N_r, L, N_c)$ configurations by their optimal $\Sigma$ values.
