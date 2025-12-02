# Integer participation and structural noise in superconducting clusters (Zenodo package)

This archive contains the code, data and scripts for the paper:

> C. Agostino, *Integer participation and structural noise in superconducting clusters* (2025).

All paths below are relative to the root of this Zenodo package (`doft-study02-superconductors-zenodo-v1/`).

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python >=3.10 recommended (tested on 3.13). `openpyxl` is required for XLSX digests.

## One-shot pipeline (configs → noise → simulator → participation → figs → validation)

From the package root:

```bash
python3 notebook/run_all_pipeline.py \
  --results-root data/raw/fingerprint-run2-v7/results_w800_p7919 \
  --materials-csv data/raw/materials_clusters_real_v7.csv \
  --tag fp_kappa_w800_p7919 \
  --materials all \
  --output-root data/processed/run_w800_p7919-v7 \
  --bounds ratios=-0.25,0.25 deltas=-0.35,0.35 f0=12,500 \
  --huber-delta 0.02 \
  --max-evals 1200 \
  --seed 123 \
  --seed-sweep 5 \
  --fit-noise-by-category \
  --k-skin 0.05 \
  --default-delta-T 0.05 \
  --default-delta-space 0.05 \
  --pressure-ref 200 \
  --c-pressure 0.05 \
  --run-sensitivity --sensitivity-perturbations 200 --sensitivity-epsilon 0.05 \
  --run-loo \
  --participation-permutations 1000 \
  --participation-bounds 0.5 5.0 \
  --participation-lambda 0.001 \
  --plot-max-n 40 --plot-delta-cap 1.0 --plot-n-null 200 \
  --plot-family-hist SC_Binary \
  --plot-loss-families SC_Binary SC_HighPressure \
  --experimental-csv data/raw/experimental_coherence.csv
```

What it produces under `data/processed/run_w800_p7919-v7/`:
- `configs/`: simulator configs per material (`material_config_*`, `ground_truth_targets_*`, `loss_weights_*`).
- `structural_noise/`: `structural_noise_summary.csv` / `structural_noise_values.json`.
- `runs/`: simulator outputs per material.
- `digest/`: CSV/XLSX digests (simulator/noise/pressure; sensitivity if enabled; LOO if enabled).
- `digest/participation_v4/`: integer-participation outputs and figures.
- `digest/participation_v4/validation/`: coherence-length validation table and plot.

## Integer participation only (if you already ran the core pipeline)

```bash
python3 notebook/tools/compute_integer_participation.py \
  --materials-csv data/raw/materials_clusters_real_v7.csv \
  --noise-csv data/processed/run_w800_p7919-v7/structural_noise/structural_noise_summary.csv \
  --mode global per_family \
  --hypotheses Fm Fm_div_2 \
  --n-permutations 1000 \
  --bounds 0.5 5.0 \
  --lambda-penalty 0.001 \
  --output-dir data/processed/run_w800_p7919-v7/digest/participation_v4 \
  --seed 123
```

Outputs: `participation_summary.csv` and `participation_manifest.json`.

## Figures (main + supplement + extras)

```bash
python3 notebook/tools/plot_integer_participation.py \
  --participation-csv data/processed/run_w800_p7919-v7/digest/participation_v4/participation_summary.csv \
  --materials-csv data/raw/materials_clusters_real_v7.csv \
  --noise-csv data/processed/run_w800_p7919-v7/structural_noise/structural_noise_summary.csv \
  --output-dir data/processed/run_w800_p7919-v7/digest/participation_v4/figures \
  --max-n-plot 40 \
  --delta-cap 1.0 \
  --n-null 200 \
  --family-hist SC_Binary \
  --loss-families SC_Binary SC_HighPressure \
  --lambda-penalty 0.001 \
  --bounds 0.5 5.0 \
  --seed 123
```

Generated files:
- Main: `fig01a_hist_N_real_vs_shuffle.png`, `fig01b_hist_delta_real_vs_shuffle.png`, `fig02_delta_by_family.png`, `fig03a_delta_vs_noise_scatter.png`, `fig03b_noise_almost_integer_vs_rest.png`, `fig04a_fbase_by_family.png`.
- Supplement: `figS01_noise_by_family.png`, `figS02_Tc_vs_Tc_ideal.png`.
- Extras: `fig_integer_hist_<family>.png` (real vs null |delta| for a chosen family), `fig_delta_vs_zxi.png` (delta vs noise), `fig_Lk_vs_f.png` (loss curves), `table_dataset_summary.csv`.

## Coherence-length validation (power-law N vs ξ0)

```bash
python3 notebook/tools/validation_coherence.py \
  --participation-csv data/processed/run_w800_p7919-v7/digest/participation_v4/participation_summary.csv \
  --experimental-csv data/raw/experimental_coherence.csv \
  --output-dir data/processed/run_w800_p7919-v7/digest/participation_v4/validation \
  --seed 123
```

Outputs: `validation_coherence_stats.csv` (Material, Family, N_model, Xi0_exp, Reference) and `validation_coherence_N.png` (log–log scatter with fitted trend).

## Repository layout (Zenodo)

- `notebook/run_all_pipeline.py`: orchestrates full pipeline for this package.
- `notebook/compute_structural_noise.py`: structural-noise calibration.
- `notebook/tools/compute_integer_participation.py`: integer participation calibration/nulls.
- `notebook/tools/plot_integer_participation.py`: figures.
- `notebook/tools/validation_coherence.py`: coherence-length validation.
- `notebook/tools/sensitivity_analysis.py`, `notebook/tools/loo_validation.py`: sensitivity/LOO.
- `notebook/doft_cluster_simulator/`: simulator core.
- `data/raw/`: input CSVs (materials, fingerprints, experimental coherence).
- `paper/`: manuscript sources.

Notes:
- All scripts assume execution from the package root; `notebook/` is added to `PYTHONPATH` internally by `run_all_pipeline.py`.
- For a lean run (skip participation/plots/validation), add `--skip-participation` to the pipeline command.

