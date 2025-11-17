# doft-study02-structural-oscillator

Code and data for DOFT Study 02: structural oscillator models for layered systems and noise propagation.

## Cómo correr el pipeline (desde cero)

1) **Generar los JSON de entrada** a partir de los resultados del fingerprint (run2-v6):

```bash
python3 src/tools/generate_doft_configs.py \
  --results-root data/raw/fingerprint-run2-v6/results_w800_p7919 \
  --tag fp_kappa_w800_p7919 \
  --materials Al Hf Mo Nb NbN Re Ta Ti V Zr \
  --output-dir configs/typeII \
  --q-strategy-single gating \
  --eta data/raw/fingerprint-run2-v6/results_w800_p7919/calib/calibration_metadata_calib_w800_p7919.json
```

Notas:
- `--eta` acepta tanto el formato nuevo (`{"eta": {"mean": ...}}`) como el legacy (`{"CALIBRATED_ETA": ...}`).
- El script crea (por material) `material_config_<mat>.json`, `ground_truth_targets_<mat>.json` y `loss_weights_default_<mat>.json`. Los contrastes `_vs_` se incluyen en `ground_truth_targets`.

2) **Ejecutar el simulador** con los JSON generados:

```bash
python3 -m scripts.doft_cluster_simulator.cli \
  --config configs/typeII/material_config_Al.json \
  --targets configs/typeII/ground_truth_targets_Al.json \
  --weights configs/typeII/loss_weights_default_Al.json \
  --bounds ratios=-0.25,0.25 deltas=-0.35,0.35 f0=12,500 \
  --huber-delta 0.02 \
  --max-evals 1200 \
  --seed 123 \
  --seed-sweep 20 \
  --outdir runs/al_unforced_f0
```

Detalles del CLI:
- `--bounds` limita el muestreo de ratios/deltas/f0 (se clampa si se pasa el rango).
- `--huber-delta` activa pérdida Huber (si se omite usa L2 cuadrática).
- `--seed-sweep` corre varias semillas en subcarpetas (`seed_<n>`); `sweep_manifest.json` referencia la mejor.
- `--anchor-weight` permite sobreescribir `w_anchor` rápido.

Los resultados de cada corrida incluyen `best_params.json`, `simulation_results.csv`, `report.md` y `manifest.json`.
