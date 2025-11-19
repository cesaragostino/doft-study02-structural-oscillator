# STUDY 02: Structural Noise Topology and Vector Dissipation in Superconducting Clusters

**Author:** Cesar Agostino  
**Date:** November 2025  
**Reference:** DOFT‑STUDY‑02  
**Precedent:** STUDY 01 – Mother Frequency & Thermal Memory Shift

## Abstract

**Study 01** showed that the critical temperature of a superconductor arises from an **oscillatory synchronisation** between electrons and the lattice. A "Mother Frequency" ($F_m$) governs the temporal coherence, while a **thermal memory** ($M_{th}$) modulates the response. When this framework was extended to multi‑band materials and high‑pressure hydrides, a systematic residue in the predicted frequencies appeared. We initially referred to it as **noise**, but careful analysis revealed that it is a physical, deterministic quantity: a **structural tension** ($\xi$) induced by the mismatch of prime‑indexed sublattices.

This report (**Study 02**) presents a vector model that decomposes this structural tension across the discrete layers of a cluster. Each layer is labelled by a prime number $p \in \{2,3,5,7\}$ corresponding to the fundamental "locks" identified in the Discrete Oscillator Fingerprint Topology (DOFT). We show that the dissipation of $\xi$ exhibits a family‑dependent topology: **binary superconductors** dissipate tension at the surface (skin), whereas **iron‑based superconductors** retain a strong core coupling. We also introduce a **pressure vector** to distinguish true structural tension from geometric compression in hydrides. Leave‑one‑out cross‑validation demonstrates that the vector model is predictive and robust, not a fitting artefact.

---

## 1. Introduction

### 1.1. Legacy of Study 01
In Study 01, we derived a **Mother Frequency** ($F_m$) and showed that the critical temperature ($T_c$) is the result of an oscillatory synchronisation of electrons and lattice. The response of the system is modulated by a **thermal memory coefficient** ($M_{th}$), which captures how the lattice retains information about prior thermal states. This formalism explained deviations in conventional and single‑band superconductors and provided a predictive link between $F_m$, the superconducting gap $\Delta$, and characteristic phonon scales $\omega_{log}$ and $\omega_{Debye}$.

### 1.2. Emergence of Structural Noise
When we applied the DOFT framework to complex materials (multi‑band compounds such as $MgB_2$, iron‑based pnictides like $FeSe$, and high‑pressure hydrides such as $LaH_{10}$), we noticed a systematic discrepancy between the predicted and measured frequencies. This **structural noise** ($\xi$) is not random; it is the energy stored in the "friction" between sublattices (e.g., $\sigma$ and $\pi$ bands) that share the same $T_c$ but differ in stiffness and discrete prime locks.

Our initial attempt to model $\xi$ with **scalar corrections**—a uniform thermal shift $\delta T$ and a uniform spatial expansion $\delta S$—failed. The optimiser collapsed these scalars to zero, indicating that the tension is neither purely thermodynamic nor homogeneous across the cluster.

## 2. Theoretical Framework

### 2.1. Discrete Layers and Prime Locks
In the DOFT approach, each superconducting cluster is decomposed into discrete layers labelled by prime numbers $p \in \{2,3,5,7\}$. These primes correspond to **lock families** (see Appendix A for details) and encode how the superconductor’s phonons and electronic states couple at different depths. A mismatch between the prime‑indexed layers of two sublattices (for example, between $\sigma$ and $\pi$ bands) generates a structural tension $\xi$.

### 2.2. Vector Parametrisation
To capture the locality of this tension, we introduce **vector corrections** for each sublattice $i$:

$$
\delta T_i = [\delta T_{i,2}, \delta T_{i,3}, \delta T_{i,5}, \delta T_{i,7}]
$$
$$
\delta S_i = [\delta S_{i,2}, \delta S_{i,3}, \delta S_{i,5}, \delta S_{i,7}]
$$

These vectors represent the thermal and spatial gradients at each layer. We also define a **pressure vector** $\delta P$ to handle externally applied pressures (in GPa), normalised by a reference pressure $P_0$ and scaled by a coefficient $k_p$. The effective residual for layer $p$ of sublattice $i$ becomes:

$$
R_{i,p} = (Obs_{i,p} - Pred_{i,p}) - \xi_{base} - (\lambda_{i,p} \cdot \delta T_{i,p}) - \delta P_{i,p}
$$

Where $\xi_{base}$ is the **base structural shift** and $\lambda_{i,p}$ is a **hybrid sensitivity matrix** that weights the contribution of each vector component according to the **band stiffness** and the **geometric depth** of the layer. A typical choice is $\lambda_{geo} = [1.0, 0.8, 0.5, 0.2]$ (skin to core) and $\lambda_{band}$ for stiff ($\approx 1.0$) and soft ($\approx 0.5$) bands, such that $\lambda_{i,p} = \lambda_{geo,p} \cdot \lambda_{band,i}$. The pressure sensitivity can be calibrated similarly.

### 2.3. Interpreting the Vectors
* **$\delta T$ (Thermal Gradient):** Captures how much the effective temperature differs from the bulk at each layer; positive values indicate that the skin is hotter relative to the core.
* **$\delta S$ (Spatial Expansion):** Represents **geometric expansion** of the cluster; a positive $\delta S$ means that layer $p$ of sublattice $i$ expands outward relative to the reference configuration.
* **$\delta P$ (Pressure Vector):** Compensates for **anisotropic compression** under external pressure. High‑pressure hydrides (e.g., $LaH_{10}$ at ~170 GPa) require this term to avoid misattributing their large noise to band mismatch.

### 2.4. Optimisation and Validation
We optimise all vector components $\delta T$, $\delta S$, $\delta P$, the scalar $\xi_{base}$ and the sensitivity coefficients using genetic algorithms. In total, twenty‑one free parameters are tuned for each material. To verify that the complexity is justified, we perform a **leave‑one‑out cross‑validation (LOO)**: the model is trained on all but one material and then used to predict the vector profile for the held‑out material. We also calculate information criteria (AIC/BIC) to ensure that the reduction in residual error outweighs the penalty for additional parameters.

## 3. Results

### 3.1. Family‑Dependent Noise Topology
Analysing a dataset of ~120 materials (including single‑band, multi‑band and high‑pressure compounds), we observe two distinct dissipation patterns:

1.  **Surface Dissipation (SC_Binary family):** Binary superconductors such as $MgB_2$ exhibit large thermal gradients at the skin ($p=2$) that decay rapidly towards the core ($p=7$). This implies that their structural tension is expelled to the surface and does not perturb the core. The correlation between total predicted noise and $\xi_{skin}$ is strong, whereas the correlation with $\xi_{core}$ is weak.
2.  **Core Coupling (SC_IronBased family):** Iron‑based materials (e.g., $FeSe$, $BaFe_2(As_{1-x}P_x)_2$) display modest gradients in the skin but a strong correlation between the noise and the core layer $p=7$. This suggests that their core remains coupled and any mismatch in stiffness propagates throughout the bulk. The correlation between total noise and $\xi_{core}$ can reach $R^2 \approx 0.64$, indicating that manipulating the core stiffness could reduce $\xi$.

### 3.2. Vector Pressure in Hydrides
In high‑pressure hydrides such as $LaH_{10}$, the pressure vector $\delta P$ shows **inverse behaviour**:

* **Outer layers ($p=2,3$):** Positive values (hardening / compression).
* **Inner layers ($p=5,7$):** Negative values (relative expansion).

This implies that hydrostatic pressure does not compress the cluster uniformly; instead, it induces a density gradient that hardens the surface and softens the core. When $\delta P$ is included, the base structural shift $\xi_{base}$ of $LaH_{10}$ decreases substantially, confirming that part of the apparent noise was due to compression rather than band tension.

### 3.3. Validation
Leave‑one‑out cross‑validation yields low prediction errors, confirming that the vector model generalises to unseen materials. Typical mean absolute errors (MAE) are:

| Parameter | MAE | Interpretation |
| :--- | :--- | :--- |
| **Base structural shift ($\xi_{base}$)** | 0.39 | Residual mismatch between bands. |
| **Thermal gradient vector ($\delta T$)** | 0.007 | Differences in effective temperature per layer. |
| **Spatial expansion vector ($\delta S$)** | 0.0067 | Geometric expansion per layer. |
| **Pressure vector ($\delta P$)** | ~0 | Only non‑zero for high‑pressure materials; deterministic once pressure is known. |

The low MAE for $\delta T$ and $\delta S$ and the stability of $\xi$ under input perturbations indicate that the vector approach is not overfitting. Comparisons of AIC/BIC show that the error reduction justifies the additional parameters.

## 4. Discussion

Study 02 reframes structural noise as a **topological fingerprint** rather than an error term. By mapping the dissipation of structural tension across discrete layers, we gain insight into how superconductors cope with internal mismatches.

The dichotomy between **skin dissipation** (binary compounds) and **core coupling** (iron‑based materials) has practical implications: designing materials with a more rigid core or more accommodating skin might reduce $\xi$ and thereby improve coherence. For example, doping an iron‑based superconductor to strengthen the inner layers could mitigate structural noise. Conversely, if one desires a material that isolates its core from surface perturbations, a binary‑type architecture is preferable.

Including the **pressure vector** not only corrects for geometric compression in hydrides but also suggests that extreme pressure may be used to tune the distribution of tension within the lattice. Observing a positive $\delta P_{skin}$ and negative $\delta P_{core}$ points to non‑uniform compression, which could be exploited to engineer new high‑$T_c$ phases.

## 5. Conclusions

This study demonstrates that the structural noise $\xi$ in complex superconductors is **vectorial and topologically organised**. By introducing layer‑specific thermal, spatial and pressure gradients, we:

1.  **Identify** two distinct dissipation mechanisms (surface vs. core) tied to material families.
2.  **Correct** for geometric deformations in high‑pressure materials using pressure vectors, separating true band tension from compression effects.
3.  **Validate** the model’s predictive power via cross‑validation and information criteria, showing that the discrete geometry of DOFT captures the essential physics.

These findings hint that improving $T_c$ may depend not only on coupling strength (as shown in Study 01) but also on a material’s **topological capacity** to manage vector tension gradients without disrupting core coherence. The vector model paves the way for designing superconductors with tailored dissipation pathways.

---

### Appendix A: Prime Locks and Structural Tension
In DOFT, prime numbers $p \in \{2,3,5,7\}$ label discrete "locks" in the superconducting cluster. Each lock corresponds to a scaling between critical temperature, gap, Debye temperature and Fermi energy: for instance, $p=2$ relates $T_c$ to $\Delta$, $p=3$ links $\Delta$ to $\omega_{log}$, $p=5$ connects $\omega_{log}$ to $\omega_{Debye}$, and $p=7$ is associated with deeper electronic modes. Mismatches of these locks across sublattices manifest as structural tension $\xi$. The vector model distributes this tension across the locks, revealing how different families of superconductors dissipate it.