# SPEC v0.2 – DOFT Cluster Simulator

## 0) Objetivo de esta versión

Cerrar el loop **MgB₂ (σ/π)** con:

- Manejo formal de `q_exp` faltante (σ) sin NaN.
- Regularización suave a anclas (`f0`).
- Control de complejidad por primos (ablations y “freeze”).
- Mejoras de robustez (early stopping, clipping, bounds).
- Seed sweep (N semillas) y reporte de dispersión.
- Métrica explícita para contraste `C_AB` y consistencia `e/q/residual`.

---

## 1) Cambios en la PÉRDIDA (Loss)

### 1.1 Definiciones

Para cada subred `s ∈ {σ, π, …}`:

**Objetivos empíricos:**

- `e_s_exp = [e2, e3, e5, e7]` (cuando aplique)
- `q_s_exp` (opcional; puede faltar)
- `r_s_exp` (residual log, opcional)

**Predicción del simulador:**

- `e_s_sim`
- `q_s_sim`
- `r_s_sim`
- `f0_s` (frecuencia base simulada)

**Ancla (opcional):** `f0_s_anchor` (numérico)

**Para cada par de subredes (contraste)** `(A,B)` con objetivo empírico:

- `C_AB_exp` y `C_AB_sim`

### 1.2 Términos de pérdida

$$
L = \sum_s ( w_e ⋅ Huber(e_s^{sim} - e_s^{exp}) + w_q(s) ⋅ Huber(q_s^{sim} - q_s^{exp}) + w_r ⋅ Huber(r_s^{sim} - r_s^{exp}) + w_{anchor} ⋅ (f0_s - f0_s^{anchor})^2 ) + \sum_{(A,B)} w_c ⋅ (C_{AB}^{sim} - C_{AB}^{exp})^2 + λ ⋅ Ω(θ)
$$

**Detalles:**

- *Huber*: `delta=0.02` (robusto a outliers en e/q/r).
- `w_q(s)`: override por subred (ver 2.2). Para σ en MgB₂, fijar `w_q(σ)=0` hasta tener `q_exp` no-NaN.
- `Ω(θ)`: regularización L2 sobre parámetros `{r2,r3,r5,r7,d2,d3,d5,d7}` con peso `λ` (bajo: `1e-4–1e-3`).
- `w_anchor` chico (0.02–0.05): estabiliza `f0` contra el ancla sin forzarlo.

---

## 2) Configuración y JSONs

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

**Nota:** `q_exp: null` en σ activa `mask_q=1` internamente ⇒ `w_q(σ)=0`.

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

## 3) CLI y flags nuevas

- `--freeze-primes 7` → congela `r7,d7=0` (se pueden pasar múltiples: `--freeze-primes 7 5`).
- `--seed-sweep 5` → corre N semillas y guarda estadísticos (mean/std/best).
- `--ablation p=2,3,5|2,3,5,7` → lanza corridas con subconjuntos de primos y compara métricas.
- `--anchor-weight 0.05` → permite ajustar `w_anchor`.
- `--bounds ratios=-0.25,0.25 deltas=-0.35,0.35` → overwrite rápido de límites.

---

## 4) Algoritmo (entrenamiento)

- **Init:** `f0 := ancla (si existe)`; `r_p=d_p=0`; respeta `freeze_primes`.
- **Semilla reproducible.**
- **Optimizador:** Adam, `lr=1e-2` → reduce on plateau (factor 0.5, patience 20, min_lr=1e-4).
- **Clipping:** grad clip a 1.0.
- **Early Stopping:** monitorear pérdida validación (si no hay split, usar “running best”), paciencia=50, delta mínima=1e-5.
- **Bounds:** proyectar parámetros tras cada step a sus cotas (`ratios/deltas/f0`).
- **Loss:** Huber en e/q/r; MSE en contraste y ancla; L2 en parámetros.
- **C_AB:** calcular explícito y penalizar con `w_c`.

---

## 5) Métricas y reportes

**Por subred:**

- `|e_sim - e_exp|` (L1 por componente y promedio)
- `|q_sim - q_exp|` si existe `q_exp`, o “NA” si está enmascarado
- `|r_sim - r_exp|`
- `|f0 - f0_anchor|`

**Contrastes:** `|C_AB_sim - C_AB_exp|`

**Global:** pérdida total; breakdown por término.

**Seed sweep:** `mean±std` de las métricas clave (`f0`, `C_AB`, `e_avg`, `q`, `r`).

**Ablations:** tabla comparando `{2,3,5}` vs `{2,3,5,7}` en error total y `|C_AB_sim - C_AB_exp|`.

**Formato de salida:**

- `best_params.json`: `f0/r_p/d_p` por subred.
- `simulation_results.csv`: métricas por seed/subconjunto primes.
- `report.md`: resumen (tablas + bullet points).
- `manifest.json`: versiones, fecha, flags usados.

---

## 6) Control de error / robustez

- Huber en e/q/r (`delta=0.02`) → estabiliza outliers.
- `w_anchor` pequeño para “pegar” `f0` a ancla sin rigidizar.
- Regularización L2 leve (`λ∈[5e−4,1e−3]`).
- Bounds estrictos en ratios/deltas/f0.
- Freeze primes para pruebas escalonadas.
- Seed sweep para estimar estabilidad.
- Ablations para medir relevancia de cada primo.
- Fallback si `q_exp == null`: `w_q=0` solo en esa subred (no usar proxies).

---

## 7) Test plan mínimo

**MgB₂ (σ/π):**

- `--freeze-primes 7`: verificar que `C_AB` y `e_exp` se sostienen.
- Reintroducir 7: debe bajar pérdida total levemente sin romper `C_AB`.
- `--seed-sweep 5`: baja varianza en `f0` y `C_AB`.

**2H-NbSe₂ (σ/π):**

- Validar que el simulador colapse hacia `C_AB ~ 0`, manteniendo `e` razonables.

**Sanity:**

Si `q_exp=null` y no se setea override ⇒ el código debe bajar `w_q` automáticamente a 0 para esa subred, sin NaNs.

---

## 8) Pseudocódigo (puntos clave)

```python
# mask por subred cuando falta q_exp
w_q_s = base_w_q
if q_exp[s] is None:
    w_q_s = 0.0

# loss por subred
L_e = huber(e_sim[s] - e_exp[s], delta=0.02).mean()
L_q = 0.0 if q_exp[s] is None else huber(q_sim[s] - q_exp[s], delta=0.02)
L_r = huber(r_sim[s] - r_exp[s], delta=0.02)
L_anchor = (f0_s - f0_anchor_s)**2 if f0_anchor_s is not None else 0.0

L_s = w_e*L_e + w_q_s*L_q + w_r*L_r + w_anchor*L_anchor

# contrastes
L_c = 0.0
for (A,B) in contrasts:
    L_c += w_c * (C_sim[A,B] - C_exp[A,B])**2

# regularización
L_reg = lambda_reg * l2_norm(params)

L_total = sum_s(L_s) + L_c + L_reg
```

---

## 9) CHANGELOG v0.2 (resumen para el repo)

- **Nuevo:** `w_q` por subred (enmascara q cuando falta), evita NaNs.
- **Nuevo:** Regularización a anclas `f0` con `w_anchor`.
- **Nuevo:** `--freeze-primes` y ablations `{2,3,5}` vs `{2,3,5,7}`.
- **Nuevo:** `--seed-sweep N` y reporte de dispersión.
- **Mejora:** Huber loss en e/q/r; gradient clipping; early stopping; bounds.
- **Reportes:** breakdown de pérdida y métricas por subred, contraste y global.

