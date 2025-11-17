# DOFT Fingerprints — Summary (final_fingerprint)

**Inputs**
- Calibration: `res_calibration/results_final_calib.csv` → `doft_config.json`
- Cluster diag: `res_clusters/results_cluster_fingerprints.csv`
- Fingerprints outdir: `res_fingerprints/` (CSV + PNG)

---

## 1) Calibration snapshot
- Condition number: **2.17e+05**
- Bootstrap (N=500):
  - **Gamma (g)**: mean = 2.31e-17, 95% CI = [2.18e-30, 3.30e-16] → effectively ~0 in metals
  - **Eta (e)**: mean = **4.34e-05**, 95% CI = **[1.01e-05, 6.81e-05]**
- LOO influence on Eta (top ±): Hf (+12.2%), Tl (+9.8%) … Zr (−6.1%)

**Takeaway:** Eta>0 robust; Gamma≈0 (as expected).

---

## 2) Cluster coefficients
- MgB2: **kappa = 5.9235e-03** (small, >0); π-channel chose **INT = 24** over RAT = 24.0000 (tie → integer).
- 2H-NbSe2: **kappa = 0**; C_AB = 0.0 (channels indistinguishable in X).

**Takeaway:** κ only where C_AB>0; π prefers integer 24 when rational tie exists.

---

## 3) Fingerprints by family

### 3.1 category → INTEGER (mean exponents of prime factors)
| category   | exp_2 | exp_3 | exp_5 | exp_7 |
|:-----------|------:|------:|------:|------:|
| SC_TypeI   | 1.56  | 0.81  | 0.52  | 0.41  |
| SC_TypeII  | 1.92  | 0.58  | 0.58  | 0.42  |
| SC_Binary  | 1.00  | 0.73  | 0.32  | 0.41  |

**Pattern:** Rigid families weight **2** (and **3**) the most → locks like 2, 6, 12, 24, 30.

### 3.2 category → RATIONAL (mean q)
| category     | mean(q) |
|:-------------|--------:|
| Superfluid   | 5.00    |
| SC_Molecular | 4.89    |
| SC_Oxide     | 4.78    |
| SC_Binary    | 5.60    |

**Pattern:** Soft/hybrid families converge to **q≈5**.

### 3.3 sub_network → INTEGER
| sub_network | exp_2 | exp_3 | exp_5 | exp_7 |
|:------------|------:|------:|------:|------:|
| pi          | 3.00  | 1.00  | 0.00  | 0.00  |
| sigma       | 0.83  | 0.67  | 0.50  | 0.33  |
| single      | 1.55  | 0.71  | 0.48  | 0.42  |

**Pattern:** **π → 24 (=2^3·3)** aparece nítido.

### 3.4 sub_network → RATIONAL (mean q)
| sub_network | mean(q) |
|:------------|--------:|
| sigma-vs-pi | 5.00    |
| pi          | 5.40    |
| single      | 5.13    |

**Pattern:** otra vez **q≈5** como invariante blando.

---

## 4) Residuals after Eta (log-space)
Mean(log R_corr_eta − log L*), sd, count:

| category     | mean  |  sd   | n  |
|:-------------|:------|:------|:---|
| SC_TypeI     | −0.043 | 0.073 | 27 |
| SC_TypeII    | −0.085 | 0.101 | 24 |
| SC_Binary    | −0.014 | 0.042 | 33 |
| SC_Oxide     | −0.037 | 0.061 |  9 |
| SC_Molecular | +0.005 | 0.010 |  9 |
| Superfluid   | −0.365 | 0.696 |  6 |

**Takeaway:** Residuals ~0 for all but **Superfluid** (small-n; heavy tails, clamp likely active).

---

## 5) Bootstrap 95% CI (highlights)
- Type I/II: exponents well constrained (see CSV: `fingerprint_final_fingerprint_bootstrap_CI95.csv`).
- RATIONAL q: **Superfluid q = 4.96 [3.33, 6.33]**, **Molecular q = 4.91 [3.22, 6.44]**, **Oxide q = 4.75 [3.67, 5.78]**.

---

## 6) Checks & next steps

**Checks passed**
- Family rules: Type I/II → integer-only; Molecular/Superfluid → rational-only; Oxide → mixed. ✔
- Eta universality: residual means near 0 (except small-N Superfluid). ✔
- κ only when C_AB>0; κ≈0 when C_AB≈0. ✔

**Actionables**
1. **Superfluid stability:** add more He/He-mix cases (or other bosonic superfluids) to tighten `q≈5` and shrink residual sd.  
2. **π-channel coverage:** include more π materials (e.g., FeSe or additional dichalcogenides if per-channel anchors exist) to validate the **24** fingerprint.  
3. **Tie-break policy:** when INT and RAT tie (e.g., 24 vs 24.0000), keep preferring **INTEGER** for π only if category/sub_network says `mixed`; otherwise prefer RAT for families labeled `rational`.  
4. **Audit export:** ensure each row logs `lock_chosen`, `prime_factorization` (or p/q), and `residual_after_eta` for reproducibility.

---

*This summary is auto-generated from your latest run (`final_fingerprint`).*
