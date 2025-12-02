# DOFT-study02-structural-oscillator

Code and data for DOFT Study 02: structural oscillator models, structural-noise calibration, and integer participation analysis.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python >=3.10 recommended (tested on 3.13). `openpyxl` is required for the XLSX digests written by the pipeline.

## End-to-end pipeline (Study 02 core)

Example run (v7 data, all materials):

```bash
python3 src/run_all_pipeline.py \
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
  --run-loo
```

Pipeline steps:
- Generate simulator configs (`material_config_*.json`, `ground_truth_targets_*.json`, `loss_weights_default_*.json`) under `<output-root>/configs`.
- Compute structural noise (`structural_noise_summary.csv` / `structural_noise_values.json`) under `<output-root>/structural_noise/` and inject `xi`, `xi_sign`, `delta_*`, and `lambda_*` into configs.
- Run the simulator per material under `<output-root>/runs/<material>/`.
- Build digests in `<output-root>/digest/` (CSV + XLSX summaries, including simulator, noise, pressure, sensitivity/LOO if enabled).

Notes:
- `--eta` is auto-resolved inside `src/run_all_pipeline.py` from `--results-root` (expects `calibration_metadata_calib_{tag}.json` in `calib/`).
- Use `--materials all` to process every material in `materials-csv`; otherwise pass an explicit list.
- Remove `--fit-noise-by-category` to use a single global zeta.

## Integer participation (Study 03 add-on)

Robust calibration and null tests:

```bash
python3 src/tools/compute_integer_participation.py \
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

Outputs:
- `participation_summary.csv` with per-material participation metrics (`N_value`, `delta_value`, `f_base_used`, z-scored noise, null-model p-values).
- `participation_manifest.json` with run metadata and winning hypothesis (Fm vs Fm/2).

## Figures (Study 03)

Use the plotting script to generate all figures (main + supplement):

```bash
python3 src/tools/plot_integer_participation.py \
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
  --bounds 0.5 5.0
```

Generated files:
- Main: `fig01a_hist_N_real_vs_shuffle.png`, `fig01b_hist_delta_real_vs_shuffle.png`, `fig02_delta_by_family.png`, `fig03a_delta_vs_noise_scatter.png`, `fig03b_noise_almost_integer_vs_rest.png`, `fig04a_fbase_by_family.png`.
- Supplement: `figS01_noise_by_family.png`, `figS02_Tc_vs_Tc_ideal.png`.
- Extras: `fig_integer_hist_<family>.png` (real vs null |delta| for a chosen family), `fig_delta_vs_zxi.png` (delta vs noise), `fig_Lk_vs_f.png` (loss curves), and `table_dataset_summary.csv` (dataset composition by family/subnet).

## Coherence-length validation (power-law regression N vs ξ0)

Match experimental coherence lengths to model participation numbers and plot the power-law fit:

```bash
python3 src/tools/validation_coherence.py \
  --participation-csv data/processed/run_w800_p7919-v7/digest/participation_v4/participation_summary.csv \
  --experimental-csv data/raw/experimental_coherence.csv \
  --output-dir data/processed/run_w800_p7919-v7/digest/participation_v4/validation \
  --seed 123
```

Outputs: `validation_coherence_stats.csv` (matched table: Material, Family, N_model, Xi0_exp, Reference) and `validation_coherence_N.png` (log–log scatter with fitted trend).

## Repository layout (key files)
- `src/run_all_pipeline.py`: end-to-end Study 02 pipeline (configs -> noise -> simulator -> digests).
- `src/compute_structural_noise.py`: structural-noise calibration (xi, delta vectors, lambda params).
- `src/tools/compute_integer_participation.py`: Study 03 integer participation calibration, null models, correlations.
- `src/tools/plot_integer_participation.py`: figures for participation vs nulls, family comparisons, noise linkage.
- `src/tools/validation_coherence.py`: matches experimental coherence lengths to participation numbers and fits N ~ xi0^alpha.
- `src/doft_cluster_simulator/`: simulator core (engine, loss, reporting).
- `data/raw/`: input CSVs (materials, fingerprints, etc.).
- `docs/`: study specs and notes.
