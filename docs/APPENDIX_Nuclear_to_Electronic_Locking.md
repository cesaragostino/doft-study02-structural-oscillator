# Appendix A — From Nuclear Modes to Electronic Locking (DOFT)

> **Scope.** This appendix specifies how to connect atomic structure to DOFT’s oscillator picture: **nuclear kernel → electronic layers → solid-state envelopes**. It is written to be “repo-friendly” (GitHub Markdown + LaTeX) and directly actionable for analysis and simulation.

---

## A.1 DOFT axioms for atoms (constructive view)

**A1.1 (Kernel):** The atomic nucleus is modeled as a *composite oscillator* with a **small set of hard collective modes** \(\{\Omega_k\}\) (e.g., breathing/monopole, giant dipole, quadrupole). Their scales vary smoothly with mass number \(A\) (empirically close to \(A^{-\alpha}\), \(\alpha\in[1/3,1/2]\)).

**A1.2 (Electronic envelopes):** Bound electrons form **resonant envelopes** \(\{\omega_j\}\) anchored by spectroscopic lines (Rydberg series, fine/hyperfine) and, in solids, by **thermal/gap/Debye/Fermi** anchors:  
\(\omega_{\rm th}=\frac{k_BT_c}{\hbar},\quad \omega_\Delta=\frac{\Delta}{\hbar},\quad \omega_D=\frac{k_B\Theta_D}{\hbar},\quad \omega_F=\frac{E_F}{\hbar}\).

**A1.3 (Locking rule):** Frequency ratios of **adjacent layers** prefer a discrete set \(\mathcal{S}_{\rm lock}\):
\[
\frac{\omega_{j+1}}{\omega_j}\in\mathcal{S}_{\rm lock}=
\begin{cases}
\textbf{Integers (products of }2,3,5,7\textbf{)} & \text{rigid families (metals, }\sigma\text{)}\\[4pt]
\textbf{Rationals }p/q\ \textbf{with } q\le 8 & \text{soft/hybrid families (}\pi\text{, molecular, superfluid)}
\end{cases}
\]

**A1.4 (Thermal–memory correction):** Detuning accumulates across layers as
\[
\frac{\Delta\omega}{\omega}\;\approx\;-\beta_\ell\,X\;-\;\eta\, d\,X\quad(\Gamma\approx 0\ \text{in metals}),
\]
with \(X=\Theta_D/T_c\) (or a local proxy), \(d\in\{1,2,3\}\) the jump “distance”, \(\beta_\ell\) a per-jump local term, and **\(\eta>0\)** the *diffusive memory* propagation (calibrated on clean metals).

**A1.5 (Compounds/multiband):** When sub-networks \(A,B\) coexist, add a **mixing/interface** term
\[
\frac{\Delta\omega}{\omega}\;\;\mathrel{+}=\;\;-\kappa\,C_{AB},\qquad C_{AB}=|X_A-X_B|,
\]
with \(\kappa\ge 0\) (least-squares fitted per compound only if drift persists).

**A1.6 (Inter-channel rule):** Ratios **between channels** (e.g., \(\sigma\) vs \(\pi\)) are evaluated by **locking only** (no \(\eta\), no \(\kappa\)): search integer \(\cup\) rational \(p/q\) (denom \(\le 8\)) that minimizes the error.

---

## A.2 Anchors and ratios

We use **adjacent-layer ratios** as observables:
- **Intra-electronic (solid):** \(R_{\rm th\to\Delta}=\omega_\Delta/\omega_{\rm th}\), \(R_{\Delta\to D}=\omega_D/\omega_\Delta\), \(R_{D\to F}=\omega_F/\omega_D\).
- **Inter-channel:** \(R_{\Delta,\sigma\to\pi}=\omega_{\Delta,\sigma}/\omega_{\Delta,\pi}\), \(R_{D,\sigma\to\pi}=\omega_{D,\sigma}/\omega_{D,\pi}\).
- **Nuclear→electronic (optional):** \(R_{n\to e}=\omega_{e,\rm low}/\Omega_{\rm nuc}\) using a representative nuclear collective mode as kernel anchor (breathing/dipole).

**Lock selection:** choose \(L^\*\in\mathcal{S}_{\rm lock}\) minimizing \(\varepsilon=\frac{|R-L^\*|}{|R|}\). Families guide \(\mathcal{S}_{\rm lock}\) (integer vs rational).

**Thermal–memory application (intra only):**
\[
R_{\rm corr} \;=\; R_{\rm obs}\cdot \max\!\bigl(\,\underline{s},\ 1-\eta\,d\,X\,\bigr)\,,
\]
with **clamp** \(\underline{s}\approx 0.2\) for extreme \(X\) (e.g., He-3).

---

## A.3 Workflow (calibrate → apply)

**Step 1 — Calibrate on clean metals (Type I/II).**  
Fit \(\eta\) (and \(\Gamma\ge 0\), typically \(\Gamma\approx 0\)) using only metallic singles; Winsorize \(X\) (e.g., \(X\le 600\)); bootstrap CIs; LOO influence.

**Step 2 — Apply to families.**  
- Intra-channel: apply \(\eta\) and select locking by family.  
- Inter-channel: locking only (no \(\eta,\kappa\)).  
- If drift remains in compounds with \(C_{AB}>0\), fit **\(\kappa\ge 0\)** via intra rows only.

**Step 3 — Validate universality.**  
Verify that the **slope of drift vs. \(d\)** approaches zero after \(\eta\); check that locking type by family (integer vs \(p/q\)) holds across materials.

---

## A.4 Minimal per-element checklist

- **Inputs** (as available): \(T_c,\ \Delta,\ \Theta_D,\ E_F\); for multiband: per-channel values and \(X_A,X_B\).  
- **Compute**: \(X=\Theta_D/T_c\), all intra/inter ratios \(R\).  
- **Apply**: intra → \(\eta\) (with clamp); inter → locking only.  
- **Decide locking** per family:  
  - metals/rigid → integers (products of \(2,3,5,7\));  
  - soft/hybrid (π, molecular, superfluid) → allow \(p/q\) (denom \(\le 8\)).  
- **Compounds**: if \(C_{AB}>0\) and drift persists, fit \(\kappa\ge 0\).  
- **Record**: \(R_{\rm obs}, R_{\rm corr}, L^\*, \varepsilon\) (before/after).

---

## A.5 Worked examples (brief)

### A.5.1 MgB\(_2\) (two-band)
Anchors (example):  
\(\omega_{\rm th}(39\,\rm K)\), \(\Delta_\sigma\approx 7.1\,\rm meV\), \(\Delta_\pi\approx 2.7\,\rm meV\), \(\Theta_{D,\sigma}\approx812\,\rm K\), \(\Theta_{D,\pi}\approx750\,\rm K\).  
- Intra–\(\sigma\): \(R_{\rm th\to\Delta}\approx 2.11 \Rightarrow \mathbf{2}\); \(R_{\Delta\to D}\approx 9.85 \Rightarrow \mathbf{10}\) (integer).  
- Intra–\(\pi\): \(R_{\rm th\to\Delta}\approx 0.80 \Rightarrow \mathbf{4/5}\); \(R_{\Delta\to D}\approx 23.9 \Rightarrow \mathbf{24}\).  
- Inter (\(\sigma/\pi\)): gaps \(\approx 2.63\) → rational \(p/q\) small denom.  
- Mixing: \(C_{AB}=|X_\sigma-X_\pi|\approx1.59\), \(\kappa\sim 6\times10^{-3}\) (small, optional).

### A.5.2 He-3 / He-4 (superfluids)
- Family: **rational \(p/q\)**; \(X\) extremely large (He-3) → apply **clamp** on \((1-\eta d X)\).  
- Expect intra ratios to prefer \(p/q\) with small denominators; inter not applicable.

---

## A.6 Practical algorithms (pseudo-code)

**Lock selection (per row):**
```text
candidates = []
if family in {integer, mixed}: candidates += integer_products(2,3,5,7, limit)
if family in {rational, mixed}: candidates += rationals_pq(max_den=8)
L* = argmin_L ( |R - L| / |R| )
```

**Intra correction:**
```text
scale = max(clamp, 1.0 - eta * d * X)   # clamp ~ 0.2
R_corr = R_obs * scale
```

**Compound κ (intra rows only, C_AB>0):**
\[
\text{Given } T_i = R_{{\rm obs},i}(1-\eta d_i X_i),\;
R_i^{(\kappa)} = T_i - \kappa\,C_{AB}\,R_{{\rm obs},i}.
\]
Least-squares with constraint \(\kappa\ge 0\):
\[
\kappa^\*=\max\!\left\{0,\ \frac{\sum_i (T_i - {\rm pv}_i)\,(C_{AB}R_{{\rm obs},i})}
{\sum_i (C_{AB}R_{{\rm obs},i})^2}\right\},
\]
where \({\rm pv}_i\) is the chosen lock value (integer or \(p/q\)).
```

---

## A.7 Defaults and guardrails

- **Calibration set:** SC\_TypeI, SC\_TypeII (singles only).  
- **Winsor X:** \(X\le 600\).  
- **Clamp:** \(\underline{s}=0.2\).  
- **Rational search:** denom \(\le 8\).  
- **Do not** apply \(\eta\) or \(\kappa\) to inter-channel ratios.  
- **Report**: before/after errors and chosen lock for auditability.

---

## A.8 Open questions / next steps

1. **Nuclear anchor choice:** use empirical collective-mode scalings for \(\Omega_{\rm nuc}\) to quantify \(R_{n\to e}\) across isotopes.  
2. **Family transitions:** map where locking flips from integer to \(p/q\) as a function of \(X\), strain, or mixing.  
3. **Mother frequency \(M\):** refine its inference by chaining locks inward (multi-layer inverse problem with priors on \(\{\Omega_k\}\)).

---

*This appendix formalizes the constructive link “nucleus → electrons → solid” within DOFT using only measured anchors and minimally parametric corrections (\(\eta,\kappa\)). It is designed to be testable, auditable, and incrementally extensible.*
