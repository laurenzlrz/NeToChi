### Parameter Estimation for the Hierarchical Stochastic Block Model

To configure the hardware mapping accurately, we need to calculate the three fundamental parameters of the hSBM: the base cluster size $n_0$, the branching factor $b$, and the base connection probability $p_0$. 

To derive these, we apply the method of moments by equating empirical graph properties with their theoretical expected values. Since there are three unknown variables, we strictly require a system of three independent mathematical estimators. 

We use the following three estimators:
1. **Mean Degree ($E[D]$):** Balances total edge density across the entire network topology.
2. **Degree Variance ($\text{Var}(D)$):** Captures the structural heterogeneity and dispersion between dense local clusters and sparse global paths.
3. **Expected Triangles ($E[T]$):** Acts as the primary proxy for local transitivity and clustering, uniquely isolating the base cluster size $n_0$.

---

#### Formulas for Mean and Variance

Let $H = \log_b(N/n_0)$ be the tree height and $\Delta N_h = n_0 b^{h-1}(b-1)$ be the number of nodes at Lowest Common Ancestor (LCA) distance $h$. 

* **Expected Mean Degree ($E[D]$):** Sums the independent connection probabilities across all tree layers:
$$E[D] = (n_0 - 1)p_0 + \sum_{h=1}^H \Delta N_h p_0 2^{-h}$$

* **Expected Degree Variance ($\text{Var}(D)$):** Sums the variances of the independent Bernoulli trials defining the edges:
$$\text{Var}(D) = (n_0 - 1)p_0(1-p_0) + \sum_{h=1}^H \Delta N_h p_0 2^{-h} (1 - p_0 2^{-h})$$

---

#### Detailed Probability Derivation of the Triangle Estimator ($E[T]$)

The expected number of triangles $E[T]$ a reference node forms is derived by summing the joint probabilities of independent edge formations across three mutually exclusive node placement cases:

**1. Intra-Cluster Configuration ($E[T_{\text{intra}}]$)**
The reference node and both external nodes reside in the identical base cluster. There are $\binom{n_0-1}{2}$ ways to select the other two nodes, and all three internal edges must form, each with probability $p_0$:
$$E[T_{\text{intra}}] = \frac{1}{2}(n_0 - 1)(n_0 - 2) p_0^3$$

**2. Mixed-Cluster Configuration ($E[T_{\text{mixed}}]$)**
The reference node and a second node are in the same base cluster, while the third node is in an external shell $h$. There are $(n_0 - 1)$ choices for the internal node and $\Delta N_h$ choices for the external node. The edge between the two internal nodes forms with probability $p_0$, while both paths to the external node drop by $2^{-h}$:
$$E[T_{\text{mixed}}] = (n_0 - 1) \sum_{h=1}^{H} \Delta N_h \cdot [p_0 \cdot (p_0 2^{-h}) \cdot (p_0 2^{-h})] = (n_0 - 1) \sum_{h=1}^{H} \Delta N_h p_0^3 4^{-h}$$

**3. Inter-Cluster Configuration ($E[T_{\text{inter}}]$)**
Both external nodes lie outside the reference node's base cluster. We split this into two basic probabilistic cases based on their shell layers:
* **Asymmetric Layers ($h_1 \neq h_2$):** One node is at distance $h_1$ and the other at $h_2$. If $h_1 < h_2$, the structural distance between the two external nodes is also forced to be $h_2$. The probability that all three edges form is $p_0 2^{-h_1} \cdot p_0 2^{-h_2} \cdot p_0 2^{-h_2} = p_0^3 2^{-(h_1 + 2h_2)}$:
$$E[T_{\text{inter, async}}] = 2 \sum_{h=2}^{H} \sum_{m=1}^{h-1} \Delta N_m \Delta N_h p_0^3 2^{-(m + 2h)}$$
* **Symmetric Layers ($h_1 = h_2 = h$):** Both nodes are in shell $h$. To close the triangle, the edge between them forms with a probability determined by their own lowest common ancestor level $h_{jk}$ within that shell:
$$E[T_{\text{inter, sync}}] = \sum_{h=1}^{H} \sum_{j < k \in \text{shell}(h)} p_0^3 2^{-(2h + h_{jk})}$$

Combining all independent cases gives the total expected triangle count:
$$E[T] = E[T_{\text{intra}}] + E[T_{\text{mixed}}] + E[T_{\text{inter, async}}] + E[T_{\text{inter, sync}}]$$

---

#### Solving the System

By extracting the empirical statistics from your graph topology ($\bar{D}, S_D^2, \bar{T}$), you construct a system of three non-linear equations where $\bar{D} = E[D]$, $S_D^2 = \text{Var}(D)$, and $\bar{T} = E[T]$. This algebraic system is solved numerically using a multi-dimensional root-finding algorithm (such as Levenberg-Marquardt or a modified Powell method) to isolate the continuous parameters $n_0$, $b$, and $p_0$.