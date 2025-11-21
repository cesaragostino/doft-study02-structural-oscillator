from __future__ import annotations

import pandas as pd

from src.tools.integer_participation import (
    calibrate_f_base,
    compute_participation_numbers,
    run_null_models,
    summarize_thresholds_by_category,
)


def _toy_dataframe(base: float = 10.0) -> pd.DataFrame:
    """Build a small synthetic dataset with clean integer participation."""

    tc = [5.0, 5.0, 5.0]
    multipliers = [1, 2, 3]
    theta = [base * t * n for t, n in zip(tc, multipliers)]
    df = pd.DataFrame({"name": ["A", "B", "C"], "ThetaD_K": theta, "Tc_K": tc})
    df["F_m"] = df["ThetaD_K"] / df["Tc_K"]
    return df


def test_calibrate_f_base_recovers_base_frequency() -> None:
    df = _toy_dataframe(base=12.5)
    result = calibrate_f_base(df, subset_indices=df.index, max_n=10, grid_size=50, refine_steps=30, return_full=True)
    assert abs(result.f_base - 12.5) < 1e-2
    assert result.subset_size == len(df)


def test_participation_metrics_match_integers() -> None:
    df = _toy_dataframe(base=8.0)
    part_df = compute_participation_numbers(df, f_base=8.0)
    assert all(part_df["N_int"] == [1.0, 2.0, 3.0])
    assert part_df["abs_delta"].sum() == 0.0


def test_null_models_produce_fractions_and_pvalues() -> None:
    df = _toy_dataframe(base=6.0)
    summary = run_null_models(
        theta_d=df["ThetaD_K"].to_numpy(dtype=float),
        tc=df["Tc_K"].to_numpy(dtype=float),
        f_base=6.0,
        thresholds=[0.02],
        permutations=5,
        seed=7,
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert 0.0 <= row["fraction_real"] <= 1.0
    assert 0.0 <= row["shuffle_p_value"] <= 1.0
    assert 0.0 <= row["continuous_p_value"] <= 1.0


def test_category_threshold_summary_counts_fractions() -> None:
    df = _toy_dataframe(base=10.0)
    df["category"] = ["catA", "catA", "catB"]
    df_part = compute_participation_numbers(df, f_base=10.0)
    summary = summarize_thresholds_by_category(df_part, thresholds=[0.02, 0.01])
    # Should have two rows, one per category
    assert set(summary["category"]) == {"catA", "catB"}
    # Fractions should be 1.0 because deltas are zero
    for col in ("fraction_below_0.02", "fraction_below_0.01"):
        assert summary[col].min() == 1.0
