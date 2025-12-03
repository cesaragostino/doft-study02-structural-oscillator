
# Study 04 – Periodic Resonance Map (extended spec)

## 0. Operational goal

Build and analyze a DOFT Resonance Periodic Table where each chemical element has a prime-vector

\[
\vec{e} = (|e_2|, |e_3|, |e_5|, |e_7|)
\]

and test whether that vector is statistically aligned with the standard electronic structure (s/p/d/f blocks), without relying on superconductivity properties or ad-hoc choices.

Main script:

```bash
python run_study04_periodic_map.py
```

---

## 1) Data inputs

### 1.1. DOFT fingerprints (you already have these)

Example base file:

```text
data/processed/config_fingerprint_summary.csv
```

Minimal column assumptions (adapt to the real schema):

- `material_id` (string)
- `material_name` (e.g. `"Nb3Sn"`, `"FeSe"`)
- `family` (e.g. `SC_Binary`, `SC_TypeII`, etc.)
- `carrier_element` (if you already defined it; otherwise it’s computed in this study)
- `e2`, `e3`, `e5`, `e7` (real-valued; signs matter in general, but in this study we will use `|e_n|`)
- optional: `N`, `Fm`, etc. (not essential here, but useful for future checks)

### 1.2. Periodic table / atomic properties

New file:

```text
data/raw/elements_resonance.json
```

Suggested structure:

```json
{
  "Pb": {
    "Z": 82,
    "block": "p",
    "group": 14,
    "period": 6,
    "mass": 207.2,
    "valence_electrons": 4
  },
  "Fe": {
    "Z": 26,
    "block": "d",
    "group": 8,
    "period": 4,
    "mass": 55.845,
    "valence_electrons": 8
  }
}
```

This can be generated via `mendeleev` / `pymatgen` and then frozen to JSON for reproducibility.

---

## 2) Step A – Selection of the “carrier element” in compounds

### 2.1. Fixed rule (no per-case tuning)

Define a function:

```python
select_carrier(material_formula, element_list)
```

Reasonable hierarchy:

1. If the compound contains at least one f-block element →  
   `carrier =` the f-block element with the highest atomic fraction.
2. Else, if it contains any d-block element →  
   `carrier =` the d-block element with the highest atomic fraction.
3. Else, if it contains any p-block element →  
   `carrier =` the heaviest p-block element (largest `Z`).
4. If all else fails →  
   `carrier =` the heaviest element in the compound.

Additional rules:

- If there is explicit doping (e.g. `La₁₋ₓYₓH₁₀`) and the dopant level is low (< 10%) → ignore the dopant for carrier selection.

Document this logic in the Study 04 README and **do not change it after looking at the results**.

### 2.2. Implementation

- Parse chemical formulas (use `pymatgen.core.Composition` or your own parser).
- For each `material_id`, compute:

  ```text
  carrier_element = select_carrier(...)
  carrier_block   = elements_resonance[carrier_element]["block"]
  ```

- Create a new intermediate CSV:

  ```text
  data/processed/element_carrier_assignments.csv
  ```

  with at least:

  ```text
  material_id, material_name, carrier_element, carrier_block, e2, e3, e5, e7
  ```

---

## 3) Step B – Aggregating fingerprints by element

Goal: move from “per material” to “per carrier element”.

- Group `element_carrier_assignments` by `carrier_element`.

For each element `E`:

- `n_materials` = number of materials where `E` is the carrier.
- Mean and median of `|e_n|`:

  - `e2_mean`, `e3_mean`, `e5_mean`, `e7_mean`
  - `e2_median`, `e3_median`, `e5_median`, `e7_median`

- Dispersion:

  - `e2_std`, `e3_std`, `e5_std`, `e7_std`

- Join with `elements_resonance.json` to obtain `block`, `Z`, `mass`, etc.

Save to:

```text
data/processed/periodic_resonance_table.csv
```

Minimal columns:

- `element`
- `block` (`s`, `p`, `d`, `f`)
- `Z`, `mass`
- `n_materials`
- `e2_mean`, `e3_mean`, `e5_mean`, `e7_mean`
- `e2_median`, `e3_median`, `e5_median`, `e7_median`

Robustness filter:

- For “strong” analyses, use only elements with `n_materials >= N_min` (e.g. 3 or 5).

---

## 4) Step C – Statistical tests vs s/p/d/f blocks

All of this goes in the script (or notebook):

```text
notebook/04_periodic_resonance_analysis.ipynb
```

### 4.1. Define the working dataset

- Load `periodic_resonance_table.csv`.
- Filter to `n_materials >= N_min`.
- Define vectors:

  \[
  \vec{e} = (|e_2|, |e_3|, |e_5|, |e_7|)
  \]

(using means or medians, depending on your chosen convention).

### 4.2. Test 1 – p vs d/f in |e₂|

Hypothesis: p-block elements have larger `|e₂|` than (d + f), and weaker higher primes.

Groups:

- `G₁ = { elements with block == "p" }`
- `G₂ = { elements with block in ["d", "f"] }`

Metrics:

- Mann–Whitney U test on `|e₂|`: `p_value_e2_p_vs_df`
- Cliff’s Δ (effect size) for `|e₂|`
- Also compare `|e₅|` and `|e₇|` in the opposite direction (expect smaller in p).

Save results to:

```text
data/processed/study04_block_stats.json
```

### 4.3. Test 2 – Higher primes in d/f vs s/p

Hypothesis: d/f blocks have `|e₅|` and `|e₇|` > s/p.

Groups:

- `G_low  = { block in ["s", "p"] }`
- `G_high = { block in ["d", "f"] }`

Metrics:

- Mann–Whitney U + Cliff’s Δ for `|e₅|`
- Mann–Whitney U + Cliff’s Δ for `|e₇|`

### 4.4. Test 3 – Block classification from primes

Simplify to two classification problems:

1. Binary: low-complexity (s + p) vs high-complexity (d + f).
2. (Optional) Multiclass s/p/d/f if enough elements are available.

Model input:

\[
(|e_2|, |e_3|, |e_5|, |e_7|)
\]

Simple models:

- Binary logistic regression, or
- k-NN with `k = 3` or `5`.

Pipeline:

- Cross-validation (e.g. leave-one-out given small sample size, or 5-fold).
- Metrics: accuracy, balanced accuracy, ROC-AUC (for the binary case).

Baseline:

- Always-majority classifier (e.g. always “s+p”).

Compare real accuracy vs baseline.

---

## 5) Step D – Null models (to avoid “mathematical artifact”)

Very important.

### 5.1. Null 1 – Permuting blocks across elements

Keep fixed:

- Prime vectors \(\vec{e}\) per element.

Randomize:

- Permute `block` labels across elements (preserving the same overall frequencies of s/p/d/f).

Repeat `N_perm` times (e.g. 5000):

- Recompute:

  - Cliff’s Δ for Tests 1 and 2.
  - Classification accuracy for Test 3.

Obtain a null distribution for each metric and compare:

- Real value vs null distribution → z-score, empirical p-value.

### 5.2. Null 2 – Random rotation of prime space (optional but strong)

Idea: is the basis `{2, 3, 5, 7}` special, or would any linear combination of 4D coordinates show correlation?

Procedure:

- Generate random invertible 4×4 matrices (e.g. random orthogonal matrices).
- Transform prime vectors:

  \[
  \vec{e}' = R \vec{e}
  \]

- Repeat the tests using \(\vec{e}'\) instead of \(\vec{e}\).

If the block correlation disappears for most rotations, that supports the idea that the “prime basis” is not a trivial artifact of an arbitrary 4D space.

---

## 6) Step E – Visualizations and final outputs

### 6.1. Periodic Resonance Table

Generate:

```text
periodic_resonance_map.png
```

Standard periodic-table layout. For each cell:

- Color = dominant prime (`argmax(|e₂|, |e₃|, |e₅|, |e₇|)`).
- Saturation = norm `||\vec{e}||` or `|e_dom|`.

Optional:

- Small “bars” for the four primes in each element cell.

### 6.2. Other figures

- Boxplots of `|e₂|`, `|e₅|`, `|e₇|` by block s/p/d/f.
- Scatter plot `|e₅|` vs `|e₂|` colored by block.
- ROC curve of the classifier (s+p vs d+f).

---

## 2. How to minimize the “mathematical artifact” criticism

Key points that both the programmer and you must respect:

### 2.1. Fixed rules, no post-hoc tuning

- Carrier selection: define it once and **do not touch it** “to make things fit”.
- Tests: predefine 3–4 key hypotheses; do not run 20 variants until you get a small p-value.

### 2.2. Explicit null models

- Permutation of blocks across elements.
- (Optional) Random rotations of the prime space.

This directly addresses the criticism: “it’s just an artifact of how the model is constructed”.

### 2.3. Report effect sizes, not just p-values

- Report Cliff’s Δ and accuracy vs baseline.
- Statements like `Δ ≈ 0.4 with p_emp ≈ 0.01` are much more solid than “p = 0.04 with small N”.

### 2.4. Control for obvious confounders

- Check whether correlations disappear when conditioning on family or `Z` (e.g. compare only within a given `Z` range).
- Make sure the correlation is not simply “families with a certain f_base”.

### 2.5. Be honest in the paper language

Do **not** claim “Correspondence Law proven”, but rather:

> “We find a statistically significant alignment between DOFT prime fingerprints and the s/p/d/f block structure.”

Reverse-engineering the mechanism goes into the *Discussion / Outlook* section.


NOTAS IMPORTANTES:
1. Desempate en Carrier Selection

La regla de jerarquía es buena, pero ¿qué pasa si tienes un compuesto con dos elementos del mismo bloque en igual proporción? (Ej. una aleación Nb−Ti, ambos d).

Regla: Agregar un criterio de desempate final. "A igualdad de jerarquía y fracción, gana el mayor Z (masa)." (El elemento más pesado suele dictar la Θ 
D
​	
  y la inercia).

2. Manejo de Outliers (N 
min
​	
 )

En el Paso B, sugieres filtrar por n_materials >= N_min.

Recomendación: Sé explícito con el valor. Sugiero N 
min
​	
 =2 o 3.

Razón: Hay muchos elementos que aparecen solo una vez en la base de datos (quizás un dopante raro). Si los incluyes, van a meter ruido. Queremos elementos que aparezcan en al menos 2 o 3 compuestos distintos para promediar su "personalidad".

3. Visualización (Tabla Periódica)

Para el Paso E, en lugar de solo "color por primo dominante", pídele al programador que intente hacer "Mini-Pie Charts" o "Barras Apiladas" dentro de cada casilla del elemento.

Por qué: El Hierro no es solo 5 o 7, es una mezcla. Ver que el Pb es todo azul (e 
2
​	
 ) y el Fe es un arcoíris (e 
mix
​	
 ) visualmente mata la discusión.