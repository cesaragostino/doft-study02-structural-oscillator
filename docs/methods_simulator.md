# DOFT-Informed Cluster Oscillator Simulator (Methods)

### Goal

Simular materiales con subredes (σ, π, modos ópticos/acústicos, 0–10 bar) usando un modelo minimalista de osciladores de fase acoplados que reproduzca los observables de tu pipeline:

- **Fingerprint entero:** exponentes `{exp_a_2, exp_b_3, exp_c_5, exp_d_7}`
- **Fingerprint racional:** `q`
- **Contraste de subred:** `C_AB` (p. ej. σ vs π, 0 vs 10 bar, La-acous vs H-optic)
- **Residuo log:** `residual = log(R_corr_eta) - log(prime_value)`

> Idea práctica: calibrar con un material/subred y predecir otro manteniendo la estructura discreta DOFT (primos activos) y variando sólo parámetros continuos (acoples, dispersión, ruido, presión).

---

## 1) Modelo dinámico

### 1.1 Ecuación (Kuramoto con sesgo discreto)

\[
\dot{\theta_i} = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i) + \xi_i(t)
\]

Donde:

- $\theta_i$: fase  
- $\omega_i$: frecuencia natural  
- $K_{ij}$: acoples  
- $\xi_i$: ruido blanco ($\sigma_\mathrm{noise}^2$)

---

### 1.2 Sesgo discreto DOFT (rejilla primo-estructurada)

Para cada subred $S$:

\[
\omega(S)^* = \Omega_0 \cdot \prod_{p \in \{2,3,5,7\}} p^{e_{S,p}}, \qquad \omega_i = \omega(S)^*(1+\delta_i), \quad \delta_i \sim N(0, \sigma_\omega^2)
\]

Los exponentes $e_{S,p}$ vienen del fingerprint entero empírico.  
$\sigma_\omega$, $K$ y $\sigma_\mathrm{noise}$ controlan el grado de lock.

---

### 1.3 Capas y subredes

- **Capas:** 1–2 (opcional 3) para jerarquía simple (basta para MgB₂ / FeSe / LaH₁₀ / He-3 / He-4)
- **Subredes:** σ, π, “La-acous”, “H1-optic”, “H2-optic”, o “0/10-bar”

**Parámetros:**
- $K_\mathrm{intra}(S)$  
- $K_\mathrm{inter}(S_1, S_2)$  
- Tamaño $N_S$  
- Dispersión $\sigma_\omega(S)$

---

## 2) Observables (compatibles con tu pipeline)

- **Espectro por subred:** FFT de $\langle e^{i\theta} \rangle$ → picos $\{\hat{\omega}^k\}$
- **Fingerprint entero:** mapear $\hat{\omega}^k$ a la rejilla DOFT → `{exp_a_2, ..., exp_d_7}`
- **Fingerprint racional:** `q`
- **Contraste:** `C_AB` (σ vs π, 0 vs 10 bar, etc.)
- **Residuo log:** `log(R_corr_eta) − log(prime_value)`

Exportar CSVs con el mismo formato que tu pipeline.

---

## 3) Parametrización empírica (semillas por material)

### MgB₂ (σ vs π)
- Hechos: `C_AB ≈ 1.5897`, σ ≠ π, residuo bajo.
- Setup: 2 capas (opcional), subredes σ y π con enteros fijados por tus promedios.  
  Ajustar $K_\mathrm{inter}$ a `C_AB≈1.59`; $\sigma_\omega$ bajo.
- Validación: calibrar σ → predecir π manteniendo primos; comprobar CIs de fingerprints/q y `C_AB`.

### FeSe (σ ≈ π)
- Hecho: `C_AB ≈ 0`
- Setup: misma rejilla DOFT o $K_\mathrm{inter}$ alto + $\sigma_\omega$ algo mayor para colapsar en un patrón común.
- Chequeo: residuo y q similares, diferencias < bandas.

### LaH₁₀ (La-acous, H1-optic, H2-optic)
- Anclas X: 1.28, 3.80, 5.80.
- Setup: 3 subredes con enteros consistentes con fingerprints.  
  Ajustar $K_\mathrm{inter}$ para el par que compare el pipeline.  
  $\sigma_\omega$ algo mayor en La-acous si hace falta.

### Superfluidos (He-4 / He-3 B)
- **He-4:** 1 capa, 1 subred por presión (1 vs 10 bar).  
  Presión $P$ cambia $\omega^*$ sin activar nuevos primos; residuo bajo; q estable (~1–3.7).  
- **He-3 B:** 0 vs 10 bar con gran `C_AB` (~195). Parametrizar $\omega^*$ distinto o $K_\mathrm{inter}$ condicionado.

---

## 4) Calibración y validación

### 4.1 Calibración mínima

Fijar enteros $\{e_{2,3,5,7}\}$ por subred con promedios bootstrap actuales.  
Elegir $\Omega_0$ para alinear escala (X/frecuencias).  
Ajustar $K_\mathrm{intra}$, $K_\mathrm{inter}$, $\sigma_\omega$, $\sigma_\mathrm{noise}$ minimizando:

\[
L = w_1 \|e_\mathrm{sim} - e_\mathrm{exp}\|_1 + w_2 |q_\mathrm{sim} - q_\mathrm{exp}| + w_3 |C_{AB}^{sim} - C_{AB}^{exp}| + w_4 RMSE(\text{residuos})
\]

### 4.2 Falsación

- **Hold-out:** σ→π (MgB₂), 0→10 bar (He-4/He-3), H1→H2 (LaH₁₀)
- **Invariancia de primos bajo P:** si aparecen nuevos, hipótesis falla.
- **Estadística:** 20–50 seeds; medias y CIs de {enteros, q, C_AB, residuo}.
- **Sensibilidad:** barrer $K_\mathrm{inter}$ y $\sigma_\omega$; buscar zona donde C_AB y residuos coincidan con bandas empíricas.

---

## 5) Defaults sugeridos

| Parámetro | MgB₂ | FeSe | LaH₁₀ (por subred) | He-4 (1/10 bar) | He-3 B (0/10 bar) |
|------------|-------|------|------------------|-----------------|-------------------|
| Capas | 2 | 1–2 | 2 | 1 | 1 |
| N/subred | 200 | 200 | 150 | 200 | 200 |
| $K_{intra}$ | 0.6 | 0.6 | 0.5 | 0.5 | 0.5 |
| $K_{inter}$ | 0.15–0.25 (C_AB≈1.59) | 0.3–0.5 (C_AB≈0) | 0.05–0.2 | 0.05–0.1 | 0.1–0.2 |
| $\sigma_\omega$ | 0.01–0.02 | 0.02–0.04 | 0.02–0.05 | 0.005–0.02 | 0.01–0.03 |
| $\sigma_{noise}$ | 1e-3 | 2e-3 | 2e-3 | 5e-4 | 1e-3 |
| $\Omega_0$ | fija (4.1) | fija | fija por subred | fija por P | fija por cond. |

**Reglas rápidas:**
- ↑ $K_{inter}$ → igualiza subredes (↓ C_AB)
- ↑ $\sigma_\omega$ → ensancha picos (↑ residuo)

---

## 6) Flujo computacional

- **Config (YAML/JSON):** capas, subredes, `{e_{2,3,5,7}}`, $K$, $N$, $\sigma$, protocolo $P$  
- **Integración:** Euler / RK4, $10^4$ pasos, $\Delta t$ chico  
- **Extracción:** FFT por subred → picos → CSV compatible  
- **Pipeline actual:** fingerprints, q, C_AB, residuo, bootstrap, tests  
- **Comparador:** tablas de error sim vs empírico  
- **Ajuste:** grid search + descenso local de $L$

---

## 7) Ejercicios de predicción

- **MgB₂ σ→π:** mantener primos; C_AB≈1.59; enteros y q dentro de CIs.  
- **He-4 0→10 bar:** mismos primos; deriva suave; q estable; residuo bajo.  
- **LaH₁₀ H1→H2:** mismos primos; ajustar $K_{inter}$; respetar saltos.  
- **FeSe σ⇄π:** $K_{inter}$ alto ⇒ C_AB→0 y fingerprints coincidentes.

---

## 8) Qué sería publicable (fuerte)

- Predicciones cruzadas (subred/condición) sin recalibrar primos, con CIs simuladas que solapen las empíricas.  
- Estabilidad de q por familia (p. ej. High-Pressure ~ 5.85 ± 0.19) sólo variando parámetros continuos.  
- Residuos por familia en bandas (MgB₂ y superfluidos).  
- Tests (KW, MWU, KS) sim vs real con p-values y tamaños de efecto consistentes.

---

## 9) Notas prácticas

- Empezar con 2 capas; agregar 3 sólo si no alcanza para clavar C_AB y residuos.  
- Superfluidos: conservar set de primos y mover $\omega^*$ con presión.  
- LaH₁₀: especificar explícitamente qué par de subredes se compara (el pipeline a veces ignora la tercera).

---

### Resumen corto

La rejilla DOFT (primos 2–3–5–7) fija la estructura discreta; los acoples y dispersión ajustan el lock observado.  
Con 1–2 capas + subredes se puede replicar q, C_AB, enteros y residuo sin “caos perfecto”.  
**Métrica clave:** mantener primos y ajustar continuo; si pide nuevos primos, la hipótesis falla (y eso también es resultado).
