# DOFT-Informed Cluster Oscillator Simulator (Methods)

### Goal

Simulate materials with subnetworks (σ, π, optical/acoustic modes, 0–10 bar) using a minimalist model of coupled phase oscillators that reproduces the observables of your pipeline:

* **Integer fingerprint:** exponents `{exp_a_2, exp_b_3, exp_c_5, exp_d_7}`
* **Rational fingerprint:** `q`
* **Subnetwork contrast:** `C_AB` (e.g. σ vs π, 0 vs 10 bar, La-acous vs H-optic)
* **Log residual:** `residual = log(R_corr_eta) - log(prime_value)`

> Practical idea: calibrate with one material/subnetwork and predict another while keeping the discrete DOFT structure (active primes) fixed and varying only continuous parameters (couplings, dispersion, noise, pressure).

---

## 1) Dynamical model

### 1.1 Equation (Kuramoto with discrete bias)

[
\dot{\theta_i} = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i) + \xi_i(t)
]

Where:

* $\theta_i$: phase
* $\omega_i$: natural frequency
* $K_{ij}$: couplings
* $\xi_i$: white noise ($\sigma_\mathrm{noise}^2$)

---

### 1.2 DOFT discrete bias (prime-structured grid)

For each subnetwork $S$:

[
\omega(S)^* = \Omega_0 \cdot \prod_{p \in {2,3,5,7}} p^{e_{S,p}}, \qquad \omega_i = \omega(S)^*(1+\delta_i), \quad \delta_i \sim N(0, \sigma_\omega^2)
]

The exponents $e_{S,p}$ come from the empirical integer fingerprint.
$\sigma_\omega$, $K$ and $\sigma_\mathrm{noise}$ control the degree of locking.

---

### 1.3 Layers and subnetworks

* **Layers:** 1–2 (optionally 3) for a simple hierarchy (enough for MgB₂ / FeSe / LaH₁₀ / He-3 / He-4)
* **Subnetworks:** σ, π, “La-acous”, “H1-optic”, “H2-optic”, or “0/10-bar”

**Parameters:**

* $K_\mathrm{intra}(S)$
* $K_\mathrm{inter}(S_1, S_2)$
* Size $N_S$
* Dispersion $\sigma_\omega(S)$

---

## 2) Observables (compatible with your pipeline)

* **Spectrum per subnetwork:** FFT of $\langle e^{i\theta} \rangle$ → peaks ${\hat{\omega}^k}$
* **Integer fingerprint:** map $\hat{\omega}^k$ to the DOFT grid → `{exp_a_2, ..., exp_d_7}`
* **Rational fingerprint:** `q`
* **Contrast:** `C_AB` (σ vs π, 0 vs 10 bar, etc.)
* **Log residual:** `log(R_corr_eta) − log(prime_value)`

Export CSVs with the same format as your pipeline.

---

## 3) Empirical parametrization (seeds per material)

### MgB₂ (σ vs π)

* Facts: `C_AB ≈ 1.5897`, σ ≠ π, low residual.
* Setup: 2 layers (optional), subnetworks σ and π with integers fixed by your averages.
  Adjust $K_\mathrm{inter}$ to `C_AB≈1.59`; low $\sigma_\omega$.
* Validation: calibrate σ → predict π while keeping primes fixed; check CIs of fingerprints/q and `C_AB`.

### FeSe (σ ≈ π)

* Fact: `C_AB ≈ 0`
* Setup: same DOFT grid or high $K_\mathrm{inter}$ + somewhat larger $\sigma_\omega$ to collapse into a common pattern.
* Check: similar residual and q, differences < bands.

### LaH₁₀ (La-acous, H1-optic, H2-optic)

* X anchors: 1.28, 3.80, 5.80.
* Setup: 3 subnetworks with integers consistent with fingerprints.
  Adjust $K_\mathrm{inter}$ for the pair compared by the pipeline.
  Slightly larger $\sigma_\omega$ in La-acous if needed.

### Superfluids (He-4 / He-3 B)

* **He-4:** 1 layer, 1 subnetwork per pressure (1 vs 10 bar).
  Pressure $P$ shifts $\omega^*$ without activating new primes; low residual; q stable (~1–3.7).
* **He-3 B:** 0 vs 10 bar with large `C_AB` (~195). Parametrize a different $\omega^*$ or a conditioned $K_\mathrm{inter}$.

---

## 4) Calibration and validation

### 4.1 Minimal calibration

Fix integers ${e_{2,3,5,7}}$ per subnetwork using current bootstrap averages.
Choose $\Omega_0$ to align the scale (X/frequencies).
Tune $K_\mathrm{intra}$, $K_\mathrm{inter}$, $\sigma_\omega$, $\sigma_\mathrm{noise}$ by minimizing:

[
L = w_1 |e_\mathrm{sim} - e_\mathrm{exp}|*1 + w_2 |q*\mathrm{sim} - q_\mathrm{exp}| + w_3 |C_{AB}^{sim} - C_{AB}^{exp}| + w_4 RMSE(\text{residuals})
]

### 4.2 Falsification

* **Hold-out:** σ→π (MgB₂), 0→10 bar (He-4/He-3), H1→H2 (LaH₁₀)
* **Prime invariance under P:** if new ones appear, the hypothesis fails.
* **Statistics:** 20–50 seeds; means and CIs of {integers, q, C_AB, residual}.
* **Sensitivity:** sweep $K_\mathrm{inter}$ and $\sigma_\omega$; search for the region where C_AB and residuals match the empirical bands.

---

## 5) Suggested defaults

| Parameter        | MgB₂                  | FeSe             | LaH₁₀ (per subnetwork) | He-4 (1/10 bar) | He-3 B (0/10 bar) |
| ---------------- | --------------------- | ---------------- | ---------------------- | --------------- | ----------------- |
| Layers           | 2                     | 1–2              | 2                      | 1               | 1                 |
| N/subnetwork     | 200                   | 200              | 150                    | 200             | 200               |
| $K_{intra}$      | 0.6                   | 0.6              | 0.5                    | 0.5             | 0.5               |
| $K_{inter}$      | 0.15–0.25 (C_AB≈1.59) | 0.3–0.5 (C_AB≈0) | 0.05–0.2               | 0.05–0.1        | 0.1–0.2           |
| $\sigma_\omega$  | 0.01–0.02             | 0.02–0.04        | 0.02–0.05              | 0.005–0.02      | 0.01–0.03         |
| $\sigma_{noise}$ | 1e-3                  | 2e-3             | 2e-3                   | 5e-4            | 1e-3              |
| $\Omega_0$       | fixed (4.1)           | fixed            | fixed per subnetwork   | fixed per P     | fixed per cond.   |

**Quick rules:**

* ↑ $K_{inter}$ → equalizes subnetworks (↓ C_AB)
* ↑ $\sigma_\omega$ → broadens peaks (↑ residual)

---

## 6) Computational flow

* **Config (YAML/JSON):** layers, subnetworks, `{e_{2,3,5,7}}`, $K$, $N$, $\sigma$, protocol $P$
* **Integration:** Euler / RK4, $10^4$ steps, small $\Delta t$
* **Extraction:** FFT per subnetwork → peaks → CSV compatible
* **Current pipeline:** fingerprints, q, C_AB, residual, bootstrap, tests
* **Comparator:** error tables sim vs empirical
* **Tuning:** grid search + local descent of $L$

---

## 7) Prediction exercises

* **MgB₂ σ→π:** keep primes; C_AB≈1.59; integers and q within CIs.
* **He-4 0→10 bar:** same primes; smooth drift; q stable; low residual.
* **LaH₁₀ H1→H2:** same primes; tune $K_{inter}$; respect jumps.
* **FeSe σ⇄π:** high $K_{inter}$ ⇒ C_AB→0 and matching fingerprints.

---

## 8) What would be publishable (strong)

* Cross predictions (subnetwork/condition) without recalibrating primes, with simulated CIs overlapping the empirical ones.
* q stability by family (e.g. High-Pressure ~ 5.85 ± 0.19) changing only continuous parameters.
* Residuals by family in bands (MgB₂ and superfluids).
* Tests (KW, MWU, KS) sim vs real with consistent p-values and effect sizes.

---

## 9) Practical notes

* Start with 2 layers; add a 3rd only if needed to nail C_AB and residuals.
* Superfluids: keep the set of primes and move $\omega^*$ with pressure.
* LaH₁₀: explicitly specify which pair of subnetworks is being compared (the pipeline sometimes ignores the third one).

---

### Short summary

The DOFT grid (primes 2–3–5–7) fixes the discrete structure; the couplings and dispersion adjust the observed locking.
With 1–2 layers + subnetworks you can replicate q, C_AB, integers and residual without “perfect chaos”.
**Key metric:** keep primes fixed and tune the continuous part; if it demands new primes, the hypothesis fails (and that is also a result).
