# STUDY 02: Structural Noise Topology and Vector Dissipation in Superconducting Clusters

**Author:** Cesar Agostino  
**Date:** November 2025  
**Reference:** DOFT-STUDY-02  
**Precedent:** STUDY 01 - Mother Frequency & Thermal Memory Shift

## Abstract

While **Study 01** established the existence of a **Mother Frequency** ($F_m$) and a **Thermal Memory** mechanism ($M_{th}$) governing temporal coherence in superconductors, a persistent residue was observed in multi-band and high-pressure materials. This residue, initially termed "noise," is not random.

The present **Study 02** demonstrates that said residue is a manifestation of **Structural Tension ($\xi$)** derived from the decoupling between sublattices. Through the implementation of a vector gradient model over prime space ($P=\{2,3,5,7\}$), we have discovered that the dissipation of this tension follows a specific topology per family: binary materials relax their tension at the surface (**Skin-Dissipation**), while iron-based superconductors maintain a deep coupling (**Core-Coupling**). Furthermore, it is revealed that extreme pressure does not act as a uniform compression, but as a deformation vector that hardens the surface and softens the core. Cross-validation (LOO) confirms that this vector model is predictive and robust, not a mathematical artifact.

---

## 1. Introduction

### 1.1. The Legacy of Study 01
In previous work (DOFT-Study01), we demonstrated that the critical temperature ($T_c$) is not a static barrier, but the result of an oscillatory synchronization governed by a fundamental frequency ($F_m$) and modulated by the system's thermal inertia ($k_{mem}$). This approach resolved most of the deviation in conventional and simple superconductors.

### 1.2. The Problem of Structural Noise
However, upon extending the DOFT (Discrete Oscillator Fingerprint Topology) model to complex materials (multi-band like $MgB_2$, iron-based like $FeSe$, or high-pressure hydrides like $LaH_{10}$), a systematic discrepancy emerged. We termed this phenomenon **Structural Noise ($\xi$)**.

The central hypothesis of this study is that $\xi$ is not a measurement error, but the energy resulting from the "friction" between sublattices (e.g., $\sigma$ vs $\pi$ bands) attempting to resonate at a shared critical temperature while occupying the same physical space but possessing different stiffnesses.

## 2. Theoretical Framework and Modeling

### 2.1. The Insufficiency of the Scalar
Our initial attempts to model this tension via a scalar thermal gradient ($\delta T_{global}$) and a scalar spatial expansion ($\delta_{space}$) resulted in null values after optimization. This indicated that tension is not resolved thermodynamically in a uniform manner between the bulk and the surface.

### 2.2. Vector Parametrization (The Topological Leap)
To capture the complexity of the phenomenon, we evolved the model towards a vector representation projected onto the discrete layers of the cluster, associated with prime numbers $p \in \{2,3,5,7\}$:

$$
\delta T_i = [\delta T_{i,2}, \delta T_{i,3}, \delta T_{i,5}, \delta T_{i,7}]
$$

Where each component represents the local thermal gradient in layer $p$ of sublattice $i$. The effective residue equation is redefined as:

$$
R_{i,p} = (Obs_{i,p} - Pred_{i,p}) - \xi_{eff} - (\lambda_{i,p} \cdot \delta T_{i,p}) - \delta P_{i,p}
$$

A **Hybrid Sensitivity Matrix** ($\lambda_{i,p} = \lambda_{band} \cdot \lambda_{geo}$) is introduced, which modulates the response according to band stiffness and geometric depth.

### 2.3. The Pressure Vector ($\delta P$)
For materials subjected to high pressures ($P > 50$ GPa), we introduce a compensation vector $\delta P$. Unlike temperature, which dilates, pressure compresses the lattice, altering the geometry of oscillatory modes in a non-linear fashion.

## 3. Results

The analysis was performed on a dataset of 120 materials, optimizing 21 free parameters using genetic algorithms and validating results with **Leave-One-Out Cross-Validation** (LOO).

### 3.1. Noise Topology by Family
The optimized vectors revealed two distinct dissipative behaviors, invisible in previous models:

* **Surface Dissipation (SC_Binary):** Materials like $MgB_2$ show significant gradients ($\delta T \approx 0.03$) at prime $p=2$ (Skin), decaying to zero at $p=7$ (Core).
    * *Interpretation:* Structural tension is "expelled" towards the surface. The core remains unaltered.
* **Core Coupling (SC_IronBased):** Materials like $FeSe$ show a strong correlation ($R^2 \approx 0.64$) between total noise and the state of the core ($p=7$), even if absolute gradients are low.
    * *Interpretation:* Core stability is critical. Tension is distributed throughout the volume; there is no "skin" to isolate internal conflict.

### 3.2. The Vector Pressure Anomaly
In high-pressure hydrides ($LaH_{10}$), the vector $\delta P$ revealed counterintuitive behavior:

* **Outer Layers ($p=2,3$):** Positive shift (Hardening/Compression).
* **Inner Layers ($p=5,7$):** Negative shift (Softening/Relative Expansion).

This suggests that external hydrostatic pressure does not compress the cluster uniformly, but generates an inverse density gradient towards the center.

### 3.3. Validation and Robustness
To rule out that vector complexity (21 parameters) was an artifact of overfitting, we subjected the model to a LOO test:

* **MAE Prediction $\delta T$:** $0.007$
* **MAE Prediction $\delta_{space}$:** $0.0067$

The model's ability to predict the exact vector profile of an unseen material confirms that the topological structure is a real and generalizable physical property.

## 4. Discussion

**Study 02** marks a paradigm shift in DOFT simulation. We have moved from considering "noise" as an error to understanding it as a **topological fingerprint**.

The distinction between **Skin-Dissipation** (Binaries) and **Core-Coupling** (Iron-Based) offers a new design tool. If we seek robust materials, the iron-type architecture (where the core participates in coherence) seems advantageous, though more prone to instabilities if decoupling $\xi$ is high. On the other hand, the binary architecture allows for "clean" superconductivity in the interior, sacrificing the surface.

Furthermore, the vector pressure correction eliminates false positives of structural noise in hydrides, demonstrating that what we interpreted as "band tension" was, in large part, a radially asymmetric geometric deformation.

## 5. Conclusion

We have demonstrated that **Structural Noise ($\xi$)** in complex superconductors possesses a predictable vector structure. The implementation of discrete gradients over prime space has allowed us to:

1.  **Identify** energy dissipation mechanisms differentiated by family (Skin vs. Core).
2.  **Correct** geometric deformations in high-pressure materials using $\delta P$ vectors.
3.  **Validate** statistically that discrete geometry (DOFT) captures the fundamental physics of the lattice.

This finding suggests that the key to raising $T_c$ lies not only in coupling strength (Study 01), but in the material's topological capacity to manage vector tension gradients without breaking core coherence.

### Appendix: Final Model Parameters

| Parameter | Type | Description | Validation (MAE) |
| :--- | :--- | :--- | :--- |
| $\xi$ | Scalar | Base Decoupling Noise | 0.39 |
| $\delta T$ | Vector [2,3,5,7] | Differential Thermal Gradient | 0.007 |
| $\delta P$ | Vector [2,3,5,7] | Response to External Pressure | N/A (Deterministic) |
| $\lambda$ | Matrix | Geo-Band Sensitivity | Calibrated by family |