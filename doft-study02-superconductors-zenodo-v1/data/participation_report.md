# Integer Participation — Study 02

- f_base (calibrated): **2949.912206** (subset n=163, loss=1.2937e+09)
- Histogram bins: 50 (total samples=8)
- First bins: [0.000, 0.010) → 0.000%, [0.010, 0.020) → 0.000%, [0.020, 0.030) → 0.000%, [0.030, 0.040) → 0.000%, [0.040, 0.050) → 0.000%
- Null models: thr<0.02: real=0.000; shuffle μ=0.002, p=1.000; cont μ=0.011, p=1.000 | thr<0.01: real=0.000; shuffle μ=0.001, p=1.000; cont μ=0.005, p=1.000 | thr<0.005: real=0.000; shuffle μ=0.001, p=1.000; cont μ=0.003, p=1.000
- Noise correlations: predicted_noise (spearman) r=0.326, p=0.029; M_struct_mean (pearson) r=0.247, p=0.102; mismatch_mean (spearman) r=0.190, p=0.211
- By-category correlations: SC_IronBased:mismatch_mean (spearman) r=0.866, p=0.000; SC_IronBased:predicted_noise (spearman) r=0.866, p=0.000; SC_IronBased:mismatch_mean (pearson) r=0.598, p=0.005
- Category thresholds: SC_Binary n=59 ⟨|δ|⟩=0.917; SC_HighPressure n=50 ⟨|δ|⟩=0.997; SC_IronBased n=25 ⟨|δ|⟩=0.995

Artifacts:
- metrics: `data/processed/run_w800_p7919-v7/digest/participation/participation_metrics.csv`
- histogram: `data/processed/run_w800_p7919-v7/digest/participation/participation_histogram.csv`
- null_models: `data/processed/run_w800_p7919-v7/digest/participation/participation_null_models.csv`
- correlations: `data/processed/run_w800_p7919-v7/digest/participation/participation_noise_correlations.csv`
- percentile_groups: `data/processed/run_w800_p7919-v7/digest/participation/participation_noise_groups.csv`
- correlations_by_category: `data/processed/run_w800_p7919-v7/digest/participation/participation_noise_correlations_by_category.csv`
- histograms_by_category: `data/processed/run_w800_p7919-v7/digest/participation/participation_histogram_by_category.csv`
- thresholds_by_category: `data/processed/run_w800_p7919-v7/digest/participation/participation_thresholds_by_category.csv`
- manifest: `data/processed/run_w800_p7919-v7/digest/participation/participation_manifest.json`