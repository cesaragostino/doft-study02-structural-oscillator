# Integer Participation — Study 02

- f_base (calibrated): **2949.912206** (subset n=163, loss=1.2937e+09)
- Histogram bins: 50 (total samples=8)
- First bins: [0.000, 0.010) → 0.000%, [0.010, 0.020) → 0.000%, [0.020, 0.030) → 0.000%, [0.030, 0.040) → 0.000%, [0.040, 0.050) → 0.000%
- Null models: thr<0.02: real=0.000; shuffle μ=0.002, p=1.000; cont μ=0.011, p=1.000 | thr<0.01: real=0.000; shuffle μ=0.001, p=1.000; cont μ=0.005, p=1.000 | thr<0.05: real=0.000; shuffle μ=0.004, p=1.000; cont μ=0.026, p=1.000
- Noise correlations: predicted_noise (spearman) r=0.326, p=0.029; M_struct_mean (pearson) r=0.247, p=0.102; mismatch_mean (spearman) r=0.190, p=0.211

Artifacts:
- metrics: `data/processed/run_w800_p7919-v7/digest/participation/participation_metrics.csv`
- histogram: `data/processed/run_w800_p7919-v7/digest/participation/participation_histogram.csv`
- null_models: `data/processed/run_w800_p7919-v7/digest/participation/participation_null_models.csv`
- correlations: `data/processed/run_w800_p7919-v7/digest/participation/participation_noise_correlations.csv`
- percentile_groups: `data/processed/run_w800_p7919-v7/digest/participation/participation_noise_groups.csv`
- manifest: `data/processed/run_w800_p7919-v7/digest/participation/participation_manifest.json`