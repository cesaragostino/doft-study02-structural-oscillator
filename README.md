# doft-study02-structural-oscillator

Code and data for DOFT Study 02: structural oscillator models for layered systems and noise propagation.

# DOFT-study02-structural-oscillator

Code and data for DOFT Study 02: structural oscillator models for layered systems and noise propagation.

## End-to-end pipeline (single command)

```bash
python3 src/run_all_pipeline.py \
  --results-root data/raw/fingerprint-run2-v6/results_w800_p7919 \
  --tag fp_kappa_w800_p7919 \
  --materials Al Hf Mo Nb NbN Re Ta Ti V Zr \
  --output-root outputs/run_w800_p7919 \
  --bounds ratios=-0.25,0.25 deltas=-0.35,0.35 f0=12,500 \
  --huber-delta 0.02 \
  --max-evals 1200 \
  --seed 123 \
  --seed-sweep 5 \
  --fit-noise-by-category
```

What it does:
- Generates configs (`material_config_*.json`, `ground_truth_targets_*.json`, `loss_weights_default_*.json`) under `<output-root>/configs`.
- Runs the simulator per material under `<output-root>/runs/<material>/`.
- Computes structural noise summary/JSON under `<output-root>/structural_noise/`.
- Produces a digest with simulator and structural-noise summaries under `<output-root>/digest/` (CSV + XLSX).

Notes:
- `--eta` is auto-resolved inside `src/run_all_pipeline.py` from the `--results-root` (expects `calibration_metadata_calib_{tag}.json` in the calib/ folder).
- Bounds and Huber delta are forwarded to the simulator. Remove `--fit-noise-by-category` to use a single global ζ.
- Use `--materials all` to process every material listed in `materials-csv`.
