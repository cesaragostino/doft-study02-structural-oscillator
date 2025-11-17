# SPEC v0.2 – DOFT Cluster Simulator

## 0) Objective of this version

Close the loop **MgB₂ (σ/π)** with:

* Formal handling of missing `q_exp` (σ) without NaN.
* Soft regularization to anchors (`f0`).
* Complexity control by primes (ablations and “freeze”).
* Robustness improvements (early stopping, clipping, bounds).
* Seed sweep (N seeds) and dispersion reporting.
* Explicit metric for contrast `C_AB` and consistency `e/q/residual`.

---

## 1) Changes in the LOSS (Loss)

### 1.1 Definitions

For each subnetwork `s ∈ {σ, π, …}`:

**Empirical targets:**

* `e_s_exp = [e2, e3, e5, e7]` (when applicable)
* `q_s_exp` (optional; may be missing)
* `r_s_exp` (log residual, optional)

**Simulator prediction:**

* `e_s_sim`
* `q_s_sim`
* `r_s_sim`
* `f0_s` (simulated base frequency)

**Anchor (optional):** `f0_s_anchor` (numeric)

**For each pair of subnetworks (contrast)** `(A,B)` with empirical target:

* `C_AB_exp` and `C_AB_sim`

### 1.2 Loss terms

$$
L = \sum_s ( w_e ⋅ Huber(e_s^{sim} - e_s^{exp}) + w_q(s) ⋅ Huber(q_s^{sim} - q_s^{exp}) + w_r ⋅ Huber(r_s^{sim} - r_s^{exp}) + w_{anchor} ⋅ (f0_s - f0_s^{anchor})^2 ) + \sum_{(A,B)} w_c ⋅ (C_{AB}^{sim} - C_{AB}^{exp})^2 + λ ⋅ Ω(θ)
$$

**Details:**

* *Huber*: `delta=0.02` (robust to outliers in e/q/r).
* `w_q(s)`: override per subnetwork (see 2.2). For σ in MgB₂, set `w_q(σ)=0` until `q_exp` is non-NaN.
* `Ω(θ)`: L2 regularization over parameters `{r2,r3,r5,r7,d2,d3,d5,d7}` with weight `λ` (low: `1e-4–1e-3`).
* `w_anchor` small (0.02–0.05): stabilizes `f0` against the anchor without forcing it.

---

## 2) Configuration and JSONs

### 2.1 `material_config.json`

```json
{
  "material": "MgB2",
  "subnetworks": ["sigma", "pi"],
  "anchors": {
    "sigma": { "f0": 20.82 },
    "pi":    { "f0": 19.23 }
  },
  "primes": [2,3,5,7],
  "constraints": {
    "ratios_bounds":  [-0.25, 0.25],
    "deltas_bounds":  [-0.35, 0.35],
    "f0_bounds":      [15.0, 30.0]
  },
  "freeze_primes": [],
  "layers": { "sigma": 1, "pi": 1 }
}
```

### 2.2 `ground_truth_targets.json`

```json
{
  "MgB2_sigma": {
    "e_exp": [0.7953, 0.6613, 0.5197, 0.312],
    "q_exp": null,
    "residual_exp": -0.008642,
    "input_exponents": [1,0,0,0]
  },
  "MgB2_pi": {
    "e_exp": [1.228, 0.7565, 0.2525, 0.489],
    "q_exp": 6.022,
    "residual_exp": -0.006469,
    "input_exponents": [3,1,0,0]
  },
  "MgB2_sigma_vs_pi": {
    "C_AB_exp": 1.58974
  }
}
```

**Note:** `q_exp: null` in σ activates `mask_q=1` internally ⇒ `w_q(σ)=0`.

### 2.3 `loss_weights_default.json`

```json
{
  "w_e": 1.0,
  "w_q": 0.5,
  "w_r": 0.25,
  "w_c": 0.3,
  "w_anchor": 0.05,
  "lambda_reg": 0.0005,
  "overrides": {
    "q": { "MgB2_sigma": 0.0 }
  }
}
```

---

## 3) CLI and new flags

* `--freeze-primes 7` → freezes `r7,d7=0` (multiple can be passed: `--freeze-primes 7 5`).
* `--seed-sweep 5` → runs N seeds and saves statistics (mean/std/best).
* `--ablation p=2,3,5|2,3,5,7` → launches runs with prime subsets and compares metrics.
* `--anchor-weight 0.05` → allows adjusting `w_anchor`.
* `--bounds ratios=-0.25,0.25 deltas=-0.35,0.35` → quick overwrite of limits.

---

## 4) Algorithm (training)

* **Init:** `f0 := anchor (if it exists)`; `r_p=d_p=0`; respect `freeze_primes`.
* **Reproducible seed.**
* **Optimizer:** Adam, `lr=1e-2` → reduce on plateau (factor 0.5, patience 20, min_lr=1e-4).
* **Clipping:** grad clip at 1.0.
* **Early Stopping:** monitor validation loss (if there is no split, use “running best”), patience=50, min delta=1e-5.
* **Bounds:** project parameters after each step back onto their bounds (`ratios/deltas/f0`).
* **Loss:** Huber on e/q/r; MSE on contrast and anchor; L2 on parameters.
* **C_AB:** compute explicitly and penalize with `w_c`.

---

## 5) Metrics and reports

**Per subnetwork:**

* `|e_sim - e_exp|` (L1 per component and average)
* `|q_sim - q_exp|` if `q_exp` exists, or “NA” if it is masked
* `|r_sim - r_exp|`
* `|f0 - f0_anchor|`

**Contrasts:** `|C_AB_sim - C_AB_exp|`

**Global:** total loss; breakdown by term.

**Seed sweep:** `mean±std` of key metrics (`f0`, `C_AB`, `e_avg`, `q`, `r`).

**Ablations:** table comparing `{2,3,5}` vs `{2,3,5,7}` in total error and `|C_AB_sim - C_AB_exp|`.

**Output format:**

* `best_params.json`: `f0/r_p/d_p` per subnetwork.
* `simulation_results.csv`: metrics per seed/prime subset.
* `report.md`: summary (tables + bullet points).
* `manifest.json`: versions, date, flags used.

---

## 6) Error control / robustness

* Huber on e/q/r (`delta=0.02`) → stabilizes outliers.
* Small `w_anchor` to “pull” `f0` toward the anchor without making it rigid.
* Light L2 regularization (`λ∈[5e−4,1e−3]`).
* Strict bounds on ratios/deltas/f0.
* Freeze primes for staged tests.
* Seed sweep to estimate stability.
* Ablations to measure the relevance of each prime.
* Fallback if `q_exp == null`: `w_q=0` only in that subnetwork (no proxies).

---

## 7) Minimal test plan

**MgB₂ (σ/π):**

* `--freeze-primes 7`: verify that `C_AB` and `e_exp` are preserved.
* Reintroduce 7: total loss should decrease slightly without breaking `C_AB`.
* `--seed-sweep 5`: low variance in `f0` and `C_AB`.

**2H-NbSe₂ (σ/π):**

* Validate that the simulator collapses toward `C_AB ~ 0`, keeping `e` reasonable.

**Sanity:**

If `q_exp=null` and no override is set ⇒ the code must automatically drop `w_q` to 0 for that subnetwork, with no NaNs.

---

## 8) Pseudocode (key points)

```python
# mask per subnetwork when q_exp is missing
w_q_s = base_w_q
if q_exp[s] is None:
    w_q_s = 0.0

# loss per subnetwork
L_e = huber(e_sim[s] - e_exp[s], delta=0.02).mean()
L_q = 0.0 if q_exp[s] is None else huber(q_sim[s] - q_exp[s], delta=0.02)
L_r = huber(r_sim[s] - r_exp[s], delta=0.02)
L_anchor = (f0_s - f0_anchor_s)**2 if f0_anchor_s is not None else 0.0

L_s = w_e*L_e + w_q_s*L_q + w_r*L_r + w_anchor*L_anchor

# contrasts
L_c = 0.0
for (A,B) in contrasts:
    L_c += w_c * (C_sim[A,B] - C_exp[A,B])**2

# regularization
L_reg = lambda_reg * l2_norm(params)

L_total = sum_s(L_s) + L_c + L_reg
```

---

## 9) CHANGELOG v0.2 (summary for the repo)

* **New:** `w_q` per subnetwork (masks q when missing), avoids NaNs.
* **New:** Regularization to `f0` anchors with `w_anchor`.
* **New:** `--freeze-primes` and ablations `{2,3,5}` vs `{2,3,5,7}`.
* **New:** `--seed-sweep N` and dispersion reporting.
* **Improved:** Huber loss in e/q/r; gradient clipping; early stopping; bounds.
* **Reports:** loss breakdown and metrics per subnetwork, contrast, and global.
