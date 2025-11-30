# Especificación Técnica: DOFT Study 04 - Atomic Resonance & Geometric Confinement

**Objetivo Principal:** Desarrollar un módulo de software que valide la conexión física del modelo DOFT en dos extremos:

* **Hacia Adentro (Micro):** Correlacionar los "Fingerprints" de primos ($e_2, e_3, e_5, e_7$) con la configuración electrónica ($s, p, d, f$) de los elementos constituyentes.
* **Hacia Afuera (Macro):** Validar matemáticamente la ley de escalamiento entre el Número de Participación ($N$) y la Longitud de Coherencia ($\xi_0$).

**Resultado Esperado:** Un modelo predictivo capaz de estimar la dimensión del par de Cooper ($\xi_0$) basado en la topología resonante ($N$) y viceversa.

---

## 1. Nuevas Fuentes de Datos

El sistema debe ingerir dos nuevos tipos de información:

### 1.1. Base de Datos de Geometría Experimental (`experimental_geometry.csv`)
Un archivo curado manualmente (o scrapeado) que contenga valores medidos para materiales clave.

* **Columnas:** `material_name`, `coherence_length_xi0_nm`, `penetration_depth_lambda_nm` (opcional), `lattice_parameter_a_nm`.
* **Uso:** *Ground truth* para validar las predicciones de DOFT.

### 1.2. Tabla Periódica de Resonancias (`elements_resonance.json`)
Un diccionario que mapea elementos químicos a sus bloques y pesos resonantes teóricos.

**Estructura y Ejemplo de datos:**
El archivo debe contener objetos donde la clave es el símbolo químico (ej. "Pb", "Fe", "La") y el valor contiene:
* "block": Orbital dominante (p, d, f).
* "valence": Valencia típica.
* "weight": Peso atómico.

Ejemplo: El Plomo (Pb) tendría bloque "p" y valencia 4; el Hierro (Fe) bloque "d" y valencia 2; el Lantano (La) bloque "f".

---

## 2. Módulo A: El Perfilador Atómico (Atomic Resonance Profiler)

**Objetivo:** Probar que la estructura interna del átomo dicta el Fingerprint DOFT.

### 2.1. Lógica de Cálculo
1.  **Parsing de Fórmula:** Leer el nombre del material (ej. `MgB2`) y descomponerlo en elementos y proporciones (Mg:1, B:2).
2.  **Determinación de Bloque Dominante:**
    * Clasificar el material según el elemento más pesado o el metal de transición presente.
    * **Jerarquía de complejidad:** $f > d > p > s$.
3.  **Correlación Cruzada:**
    * Tomar los valores $e_2, e_3, e_5, e_7$ calculados en Study 01.
    * Calcular correlación (Pearson/Spearman) entre:
        * Pertenencia al Bloque **P/S** <-> Magnitud de $|e_2|$ (Resonancia Binaria).
        * Pertenencia al Bloque **D** <-> Magnitud de $|e_5|$ (Resonancia Compleja).
        * Pertenencia al Bloque **F** <-> Magnitud de $|e_7|$ (Resonancia Profunda).

### 2.2. Output Gráfico
* **Boxplot:** Distribución de $|e_n|$ agrupada por Bloque ($s, p, d, f$).
* **Tabla Generada:** `periodic_table_resonance_map.csv` (Promedios de primos por elemento).

---

## 3. Módulo B: El Validador Geométrico (Confinement Validator)

**Objetivo:** Responder a la crítica de validación física derivando $\xi_0$ desde $N$.

### 3.1. Modelo de Regresión de Potencia
Implementar el ajuste descubierto manualmente:

$$\xi_{0,pred} = A \cdot (N_{corrected})^\beta$$

* **Entrenamiento:** Usar el subset de materiales que tengan datos en `experimental_geometry.csv`.
* **Ajuste:** Encontrar $A$ y $\beta$ (esperamos $\beta \approx 0.5$ para difusión o $\approx 1.0$ para expansión lineal).
* **Métrica de Error:** Calcular el MAPE (*Mean Absolute Percentage Error*) para definir la calidad de la predicción.
* **Criterio de Éxito:** Si MAPE < 30%, el modelo es físico.

### 3.2. Predicción Inversa (El "Game Changer")
Usar el modelo ajustado para **predecir** el $\xi_0$ de todos los materiales del dataset que no tienen datos experimentales medidos.

* **Output:** Columna `predicted_coherence_length_nm` en el reporte final.

---

## 4. Módulo C: El Simulador de "Sintonía" (Tuning Simulator)

**Objetivo:** Simular cómo cambiaría $N$ (y por ende $T_c$) si alteramos la composición atómica (Ingeniería Inversa).

### 4.1. Algoritmo de Simulación
1.  **Input:** Un material base (ej. `LaH10`).
2.  **Perturbación:** Simular la adición de un dopante (ej. reemplazar 10% de La con Y).
    * Esto altera la $\Theta_D$ efectiva (por masa) y el $f_{base}$ teórico (por cambio de bloque $f \to d$).
3.  **Cálculo de Nuevo $N$:** Recalcular $N_{new} = F_m / f_{base,mix}$.
4.  **Evaluación de Lock:**
    * Calcular distancia al entero: $d = |N_{new} - \text{round}(N_{new})|$.
    * **Hipótesis:** Si $d_{new} < d_{old}$, la aleación es más estable y podría tener mayor $T_c$.

---

## 5. Reportes y Entregables

El script `run_study04_atomic_geometric.py` debe generar:

* **`atomic_resonance_matrix.csv`:**
    * Matriz de correlación entre Orbitales ($s, p, d, f$) y Primos (2, 3, 5, 7).
    * Prueba la hipótesis del "Adentro".
* **`geometric_validation.png` & `.csv`:**
    * Scatter plot log-log de $N$ vs $\xi_0$ con la línea de ajuste y el $R^2$.
    * Tabla de predicción de $\xi_0$ para nuevos materiales.
    * Prueba la hipótesis del "Afuera" (Confinamiento).
* **`resonance_periodic_table.png`:**
    * Una visualización (heatmap) de la tabla periódica coloreada por el "Primo Dominante" de cada elemento.

### Notas para el Programador
* **Integración:** Este estudio requiere leer los outputs de Study 01 (Fingerprints) y Study 03 (Participation $N$). Asegurar que el pipeline cargue los CSVs generados anteriormente.
* **Física:** En el Módulo B, recordar que $N$ debe ser el $N_{corrected}$ (usando $F_m^*$) para que la correlación con la geometría física sea limpia.
* **Librerías:** Se recomienda usar `mendeleev` o `pymatgen` para obtener propiedades atómicas estándar (masa, configuración electrónica) automáticamente, si es posible, para no cargar todo a mano en el JSON.