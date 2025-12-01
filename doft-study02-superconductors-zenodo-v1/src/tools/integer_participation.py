"""Integer participation analysis and coupling to structural noise (Study 02).

Implements:
* Calibration of the base inertial frequency f_base on a chosen subset.
* Per-material participation metrics (N_i, delta_i, abs_delta_i, nearest magic).
* Null models (ThetaD shuffle and continuous F_m resampling) with p-values.
* Correlations between |delta| and structural noise metrics.
* Lightweight Markdown/CSV reports for digest consumption.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.02, 0.01, 0.05)
MAGIC_NUMBERS: Tuple[int, ...] = (13, 55, 147, 309, 561, 923)


@dataclass
class CalibrationResult:
    f_base: float
    loss: float
    subset_size: int
    grid_f: List[float]
    grid_loss: List[float]


def _clean_fm(df: pd.DataFrame) -> pd.Series:
    """Compute F_m = ThetaD_K / Tc_K with basic sanity checks."""

    theta = pd.to_numeric(df.get("ThetaD_K"), errors="coerce")
    tc = pd.to_numeric(df.get("Tc_K"), errors="coerce")
    fm = theta / tc
    fm[~np.isfinite(fm)] = np.nan
    fm[fm <= 0] = np.nan
    return fm


def _resolve_subset(df: pd.DataFrame, subset_indices: Optional[Iterable[int]]) -> pd.Index:
    if subset_indices is None:
        return df.index
    subset_list = list(subset_indices)
    if not subset_list:
        return df.index
    if all(isinstance(val, bool) for val in subset_list) and len(subset_list) == len(df):
        mask = pd.Series(subset_list, index=df.index)
        return df.index[mask]
    return df.index.intersection(subset_list)


def calibrate_f_base(
    data: pd.DataFrame,
    subset_indices: Optional[Iterable[int]],
    max_n: int = 500,
    grid_size: int = 400,
    refine_steps: int = 200,
    return_full: bool = False,
) -> CalibrationResult | float:
    """Calibrate f_base by minimising Σ_i (F_m,i - N_i f)^2 over a subset.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain column ``F_m`` (ΘD/Tc) with positive, finite values.
    subset_indices : iterable or None
        DataFrame indices to use for calibration; defaults to whole dataset.
    max_n : int
        Upper bound for the effective participation integer in the coarse search.
    grid_size : int
        Number of geometrically spaced samples in the first pass.
    refine_steps : int
        Number of linear samples in the refinement window around the best coarse hit.
    return_full : bool
        When True, return a CalibrationResult instead of just f_base.
    """

    if "F_m" not in data.columns:
        raise ValueError("data must contain an F_m column (ΘD/Tc).")
    subset_idx = _resolve_subset(data, subset_indices)
    fm_values = pd.to_numeric(data.loc[subset_idx, "F_m"], errors="coerce")
    fm_values = fm_values[np.isfinite(fm_values) & (fm_values > 0)]
    if fm_values.empty:
        raise ValueError("No finite/positive F_m values available for calibration subset.")

    max_n = max(int(max_n), 1)
    f_min = max(fm_values.min() / max_n, 1e-6)
    f_max = fm_values.max()
    grid = np.geomspace(f_min, f_max, num=max(grid_size, 10))

    def _loss(f: float) -> float:
        if f <= 0 or not math.isfinite(f):
            return float("inf")
        n_vals = np.round(fm_values.to_numpy(dtype=float) / f)
        n_vals = np.clip(n_vals, 1.0, float(max_n))
        residuals = fm_values.to_numpy(dtype=float) - n_vals * f
        return float(np.sum(residuals * residuals))

    coarse_losses = np.array([_loss(val) for val in grid])
    best_idx = int(np.argmin(coarse_losses))
    best_f = float(grid[best_idx])
    best_loss = float(coarse_losses[best_idx])

    lo = max(grid[max(best_idx - 1, 0)] * 0.8, f_min)
    hi = min(grid[min(best_idx + 1, len(grid) - 1)] * 1.25, f_max)
    if hi > lo:
        refine_grid = np.linspace(lo, hi, num=max(refine_steps, 10))
        refine_losses = np.array([_loss(val) for val in refine_grid])
        ref_idx = int(np.argmin(refine_losses))
        best_f = float(refine_grid[ref_idx])
        best_loss = float(refine_losses[ref_idx])
        grid = np.concatenate([grid, refine_grid])
        coarse_losses = np.concatenate([coarse_losses, refine_losses])

    result = CalibrationResult(
        f_base=best_f,
        loss=best_loss,
        subset_size=len(fm_values),
        grid_f=[float(x) for x in grid],
        grid_loss=[float(x) for x in coarse_losses],
    )
    return result if return_full else result.f_base


def compute_participation_numbers(
    df: pd.DataFrame,
    f_base: float,
    magic_numbers: Sequence[int] = MAGIC_NUMBERS,
) -> pd.DataFrame:
    """Compute N_i, delta_i, and abs_delta_i for each material."""

    result = df.copy()
    result["F_m"] = _clean_fm(result)
    if f_base <= 0 or not math.isfinite(f_base):
        raise ValueError("f_base must be a positive finite scalar.")
    n_raw = result["F_m"] / f_base
    n_int = np.maximum(1.0, np.round(n_raw))
    delta = n_raw - n_int
    result["N_raw"] = n_raw
    result["N_int"] = n_int
    result["delta"] = delta
    result["abs_delta"] = delta.abs()
    result["relative_error"] = (result["F_m"] - n_int * f_base).abs() / result["F_m"]

    if magic_numbers:
        magic_arr = np.array(magic_numbers, dtype=float)
        def _nearest_magic(val: float) -> Tuple[float, float]:
            if not math.isfinite(val):
                return float("nan"), float("nan")
            idx = int(np.argmin(np.abs(magic_arr - val)))
            nearest = magic_arr[idx]
            return float(nearest), float(abs(val - nearest))

        magic_info = n_raw.apply(_nearest_magic)
        result["nearest_magic_number"] = [pair[0] for pair in magic_info]
        result["nearest_magic_distance"] = [pair[1] for pair in magic_info]
    return result


def abs_delta_histogram(abs_delta: pd.Series, bin_width: float = 0.01, max_delta: float = 0.5) -> pd.DataFrame:
    """Histogram of |delta| with configurable bin width."""

    clean = abs_delta[np.isfinite(abs_delta)]
    clean = clean[clean <= max_delta]
    bins = np.arange(0.0, max(clean.max() if not clean.empty else max_delta, bin_width) + bin_width, bin_width)
    counts, edges = np.histogram(clean, bins=bins)
    fractions = counts / max(len(clean), 1)
    rows = []
    for idx in range(len(counts)):
        rows.append(
            {
                "bin_start": float(edges[idx]),
                "bin_end": float(edges[idx + 1]),
                "count": int(counts[idx]),
                "fraction": float(fractions[idx]),
            }
        )
    return pd.DataFrame(rows)


def abs_delta_histogram_by_category(participation_df: pd.DataFrame, bin_width: float, max_delta: float) -> pd.DataFrame:
    """Histogram of |delta| per category/family."""

    cat_col = None
    for candidate in ("category", "category_x", "category_y"):
        if candidate in participation_df.columns:
            cat_col = candidate
            break
    if cat_col is None:
        return pd.DataFrame()
    rows = []
    for category, group in participation_df.groupby(participation_df[cat_col].fillna("Unknown")):
        hist = abs_delta_histogram(group["abs_delta"], bin_width=bin_width, max_delta=max_delta)
        for entry in hist.itertuples():
            rows.append(
                {
                    "category": category,
                    "bin_start": entry.bin_start,
                    "bin_end": entry.bin_end,
                    "count": entry.count,
                    "fraction": entry.fraction,
                }
            )
    return pd.DataFrame(rows)


def _fraction_below(abs_delta: np.ndarray, thresholds: Sequence[float]) -> Dict[float, float]:
    fractions: Dict[float, float] = {}
    total = max(len(abs_delta), 1)
    for thr in thresholds:
        fractions[thr] = float(np.sum(abs_delta < thr) / total)
    return fractions


def run_null_models(
    theta_d: np.ndarray,
    tc: np.ndarray,
    f_base: float,
    thresholds: Sequence[float],
    permutations: int = 500,
    seed: int = 123,
) -> pd.DataFrame:
    """Evaluate shuffle and continuous null models."""

    rng = np.random.default_rng(seed)
    mask = np.isfinite(theta_d) & np.isfinite(tc) & (tc > 0)
    if not np.any(mask):
        raise ValueError("No valid ThetaD/Tc pairs for null model evaluation.")
    theta_clean = theta_d[mask]
    tc_clean = tc[mask]
    fm_real = theta_clean / tc_clean
    abs_delta_real = np.abs(fm_real / f_base - np.round(fm_real / f_base).clip(min=1))
    real_fracs = _fraction_below(abs_delta_real, thresholds)

    shuffle_fracs = {thr: [] for thr in thresholds}
    continuous_fracs = {thr: [] for thr in thresholds}
    fm_mean = float(np.mean(fm_real))
    fm_std = float(np.std(fm_real))

    for _ in range(max(1, permutations)):
        # Null 1: shuffle ThetaD
        perm_theta = rng.permutation(theta_clean)
        fm_perm = perm_theta / tc_clean
        abs_delta_perm = np.abs(fm_perm / f_base - np.round(fm_perm / f_base).clip(min=1))
        for thr in thresholds:
            shuffle_fracs[thr].append(float(np.sum(abs_delta_perm < thr) / len(abs_delta_perm)))

        # Null 2: continuous / resampled F_m
        fm_sample = rng.choice(fm_real, size=len(fm_real), replace=True)
        fm_continuous = rng.normal(loc=fm_mean, scale=fm_std, size=len(fm_real))
        fm_continuous = np.clip(fm_continuous, a_min=1e-6, a_max=None)
        fm_mix = rng.choice(np.concatenate([fm_sample, fm_continuous]), size=len(fm_real), replace=True)
        abs_delta_cont = np.abs(fm_mix / f_base - np.round(fm_mix / f_base).clip(min=1))
        for thr in thresholds:
            continuous_fracs[thr].append(float(np.sum(abs_delta_cont < thr) / len(abs_delta_cont)))

    rows = []
    for thr in thresholds:
        shuffle_arr = np.array(shuffle_fracs[thr], dtype=float)
        cont_arr = np.array(continuous_fracs[thr], dtype=float)
        p_shuffle = float(np.mean(shuffle_arr >= real_fracs[thr])) if shuffle_arr.size else float("nan")
        p_cont = float(np.mean(cont_arr >= real_fracs[thr])) if cont_arr.size else float("nan")
        rows.append(
            {
                "threshold": float(thr),
                "fraction_real": real_fracs[thr],
                "shuffle_fraction_mean": float(np.mean(shuffle_arr)) if shuffle_arr.size else float("nan"),
                "shuffle_fraction_std": float(np.std(shuffle_arr)) if shuffle_arr.size else float("nan"),
                "shuffle_p_value": p_shuffle,
                "continuous_fraction_mean": float(np.mean(cont_arr)) if cont_arr.size else float("nan"),
                "continuous_fraction_std": float(np.std(cont_arr)) if cont_arr.size else float("nan"),
                "continuous_p_value": p_cont,
                "permutations": permutations,
            }
        )
    return pd.DataFrame(rows)


def summarize_thresholds_by_category(participation_df: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    """Compute fraction of |delta| below thresholds per category/family."""

    cat_col = None
    for candidate in ("category", "category_x", "category_y"):
        if candidate in participation_df.columns:
            cat_col = candidate
            break
    if cat_col is None:
        return pd.DataFrame()
    thresholds = list(thresholds)
    rows = []
    for category, group in participation_df.groupby(participation_df[cat_col].fillna("Unknown")):
        abs_delta = group["abs_delta"].to_numpy(dtype=float)
        abs_delta = abs_delta[np.isfinite(abs_delta)]
        if abs_delta.size == 0:
            continue
        entry = {
            "category": category,
            "count": int(abs_delta.size),
            "abs_delta_mean": float(np.mean(abs_delta)),
            "abs_delta_median": float(np.median(abs_delta)),
        }
        for thr in thresholds:
            entry[f"fraction_below_{thr}"] = float(np.sum(abs_delta < thr) / abs_delta.size)
        rows.append(entry)
    return pd.DataFrame(rows)


def _corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> Tuple[float, float]:
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 2:
        return float("nan"), float("nan")
    if method == "spearman":
        res = stats.spearmanr(clean.iloc[:, 0], clean.iloc[:, 1])
    else:
        res = stats.pearsonr(clean.iloc[:, 0], clean.iloc[:, 1])
    return float(res.statistic), float(res.pvalue)


def correlate_with_noise(
    participation_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    metrics: Sequence[str] = ("predicted_noise", "M_struct_mean", "mismatch_mean"),
    quantile: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correlate abs_delta with available structural noise metrics."""

    merged = participation_df.merge(noise_df, left_on="name", right_on="material", how="left")
    if "abs_delta" not in merged.columns:
        raise ValueError("participation_df must contain abs_delta.")

    corr_rows = []
    group_rows = []
    for metric in metrics:
        if metric not in merged.columns:
            continue
        corr_p, pval_p = _corr(merged["abs_delta"], merged[metric], method="pearson")
        corr_s, pval_s = _corr(merged["abs_delta"], merged[metric], method="spearman")
        corr_rows.append(
            {
                "metric": metric,
                "method": "pearson",
                "coefficient": corr_p,
                "p_value": pval_p,
                "n": int(merged[["abs_delta", metric]].dropna().shape[0]),
            }
        )
        corr_rows.append(
            {
                "metric": metric,
                "method": "spearman",
                "coefficient": corr_s,
                "p_value": pval_s,
                "n": int(merged[["abs_delta", metric]].dropna().shape[0]),
            }
        )
        cutoff = merged["abs_delta"].quantile(quantile)
        low = merged[merged["abs_delta"] <= cutoff]
        rest = merged[merged["abs_delta"] > cutoff]
        group_rows.append(
            {
                "metric": metric,
                "quantile": float(quantile),
                "cutoff": float(cutoff),
                "low_mean": float(low[metric].mean()) if not low.empty else float("nan"),
                "rest_mean": float(rest[metric].mean()) if not rest.empty else float("nan"),
                "low_count": int(len(low)),
                "rest_count": int(len(rest)),
            }
        )
    return pd.DataFrame(corr_rows), pd.DataFrame(group_rows)


def correlate_with_noise_by_category(
    participation_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    metrics: Sequence[str] = ("predicted_noise", "M_struct_mean", "mismatch_mean"),
) -> pd.DataFrame:
    """Correlate |delta| with noise metrics per category/family."""

    cat_col = None
    for candidate in ("category", "category_x", "category_y"):
        if candidate in participation_df.columns:
            cat_col = candidate
            break
    if cat_col is None:
        return pd.DataFrame()
    merged = participation_df.merge(noise_df, left_on="name", right_on="material", how="left", suffixes=("_part", "_noise"))
    merged_cat = None
    for candidate in ("category_part", "category_x", "category"):
        if candidate in merged.columns:
            merged_cat = candidate
            break
    if merged_cat is None:
        merged_cat = cat_col
    rows = []
    for category, group in merged.groupby(merged[merged_cat].fillna("Unknown")):
        for metric in metrics:
            if metric not in group.columns:
                continue
            coeff_p, pval_p = _corr(group["abs_delta"], group[metric], method="pearson")
            coeff_s, pval_s = _corr(group["abs_delta"], group[metric], method="spearman")
            rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "method": "pearson",
                    "coefficient": coeff_p,
                    "p_value": pval_p,
                    "n": int(group[["abs_delta", metric]].dropna().shape[0]),
                }
            )
            rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "method": "spearman",
                    "coefficient": coeff_s,
                    "p_value": pval_s,
                    "n": int(group[["abs_delta", metric]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _render_report(
    f_base: float,
    calibration_loss: float,
    subset_size: int,
    hist_df: pd.DataFrame,
    null_df: Optional[pd.DataFrame],
    corr_df: Optional[pd.DataFrame],
    corr_cat_df: Optional[pd.DataFrame],
    cat_thresh_df: Optional[pd.DataFrame],
    out_paths: Dict[str, Path],
) -> str:
    lines = ["# Integer Participation — Study 02", ""]
    lines.append(f"- f_base (calibrated): **{f_base:.6f}** (subset n={subset_size}, loss={calibration_loss:.4e})")
    if not hist_df.empty:
        tail = hist_df.sort_values(by="bin_start").head(5)
        total = hist_df["count"].sum()
        lines.append(f"- Histogram bins: {len(hist_df)} (total samples={total})")
        pretty_rows = [f"[{row.bin_start:.3f}, {row.bin_end:.3f}) → {row.fraction:.3%}" for row in tail.itertuples()]
        lines.append(f"- First bins: {', '.join(pretty_rows)}")
    if null_df is not None and not null_df.empty:
        parts = []
        for row in null_df.itertuples():
            parts.append(
                f"thr<{row.threshold:g}: real={row.fraction_real:.3f}; "
                f"shuffle μ={row.shuffle_fraction_mean:.3f}, p={row.shuffle_p_value:.3f}; "
                f"cont μ={row.continuous_fraction_mean:.3f}, p={row.continuous_p_value:.3f}"
            )
        lines.append("- Null models: " + " | ".join(parts))
    if corr_df is not None and not corr_df.empty:
        best = corr_df.sort_values(by="coefficient", ascending=False).head(3)
        desc = [f"{row.metric} ({row.method}) r={row.coefficient:.3f}, p={row.p_value:.3f}" for row in best.itertuples()]
        lines.append("- Noise correlations: " + "; ".join(desc))
    if corr_cat_df is not None and not corr_cat_df.empty:
        top = corr_cat_df.sort_values(by="coefficient", ascending=False).head(3)
        desc_cat = [f"{row.category}:{row.metric} ({row.method}) r={row.coefficient:.3f}, p={row.p_value:.3f}" for row in top.itertuples()]
        lines.append("- By-category correlations: " + "; ".join(desc_cat))
    if cat_thresh_df is not None and not cat_thresh_df.empty:
        preview = cat_thresh_df.head(3)
        summary = [f"{row.category} n={row.count} ⟨|δ|⟩={row.abs_delta_mean:.3f}" for row in preview.itertuples()]
        lines.append("- Category thresholds: " + "; ".join(summary))
    lines.append("")
    lines.append("Artifacts:")
    for label, path in out_paths.items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines)


def run_cli(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.materials_csv)
    if args.materials:
        df = df[df["name"].isin(args.materials)]
    df["F_m"] = _clean_fm(df)
    df = df.dropna(subset=["F_m"])

    subset_indices = None
    if args.subset:
        subset_indices = df[df["name"].isin(args.subset)].index
    elif args.subset_category:
        subset_indices = df[df["category"].isin(args.subset_category)].index

    if args.f_base_override:
        f_base_result = CalibrationResult(
            f_base=float(args.f_base_override),
            loss=0.0,
            subset_size=len(subset_indices) if subset_indices is not None else len(df),
            grid_f=[],
            grid_loss=[],
        )
    else:
        f_base_result = calibrate_f_base(
            df,
            subset_indices=subset_indices,
            max_n=args.max_n,
            grid_size=args.grid_size,
            refine_steps=args.refine_steps,
            return_full=True,
        )

    participation_df = compute_participation_numbers(df, f_base=float(f_base_result.f_base))
    hist_df = abs_delta_histogram(participation_df["abs_delta"], bin_width=args.bin_width, max_delta=args.max_delta)
    thresholds = [float(x) for x in (args.thresholds or DEFAULT_THRESHOLDS)]
    null_df = None
    if args.permutations > 0:
        null_df = run_null_models(
            theta_d=participation_df["ThetaD_K"].to_numpy(dtype=float),
            tc=participation_df["Tc_K"].to_numpy(dtype=float),
            f_base=float(f_base_result.f_base),
            thresholds=thresholds,
            permutations=args.permutations,
            seed=args.seed,
        )

    corr_df = None
    group_df = None
    corr_cat_df = None
    if args.noise_csv and Path(args.noise_csv).exists():
        noise_df = pd.read_csv(args.noise_csv)
        noise_df["material"] = noise_df.get("material", noise_df.get("name"))
        corr_df, group_df = correlate_with_noise(participation_df, noise_df)
        corr_cat_df = correlate_with_noise_by_category(participation_df, noise_df)
    cat_hist_df = abs_delta_histogram_by_category(participation_df, bin_width=args.bin_width, max_delta=args.max_delta)
    cat_threshold_df = summarize_thresholds_by_category(participation_df, thresholds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "participation_metrics.csv"
    participation_df.to_csv(metrics_path, index=False)
    hist_path = out_dir / "participation_histogram.csv"
    hist_df.to_csv(hist_path, index=False)
    outputs: Dict[str, Path] = {"metrics": metrics_path, "histogram": hist_path}

    manifest = {
        "materials_csv": str(args.materials_csv),
        "noise_csv": str(args.noise_csv) if args.noise_csv else None,
        "subset": list(args.subset or []),
        "subset_category": list(args.subset_category or []),
        "f_base": float(f_base_result.f_base),
        "calibration_loss": float(f_base_result.loss),
        "subset_size": int(f_base_result.subset_size),
        "thresholds": thresholds,
        "permutations": int(args.permutations),
        "bin_width": float(args.bin_width),
        "max_delta": float(args.max_delta),
    }

    if null_df is not None:
        null_path = out_dir / "participation_null_models.csv"
        null_df.to_csv(null_path, index=False)
        outputs["null_models"] = null_path
    if corr_df is not None:
        corr_path = out_dir / "participation_noise_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        outputs["correlations"] = corr_path
    if group_df is not None:
        group_path = out_dir / "participation_noise_groups.csv"
        group_df.to_csv(group_path, index=False)
        outputs["percentile_groups"] = group_path
    if corr_cat_df is not None and not corr_cat_df.empty:
        corr_cat_path = out_dir / "participation_noise_correlations_by_category.csv"
        corr_cat_df.to_csv(corr_cat_path, index=False)
        outputs["correlations_by_category"] = corr_cat_path
    if not cat_hist_df.empty:
        cat_hist_path = out_dir / "participation_histogram_by_category.csv"
        cat_hist_df.to_csv(cat_hist_path, index=False)
        outputs["histograms_by_category"] = cat_hist_path
    if not cat_threshold_df.empty:
        cat_thresh_path = out_dir / "participation_thresholds_by_category.csv"
        cat_threshold_df.to_csv(cat_thresh_path, index=False)
        outputs["thresholds_by_category"] = cat_thresh_path

    manifest_path = out_dir / "participation_manifest.json"
    manifest["outputs"] = {k: str(v) for k, v in outputs.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    outputs["manifest"] = manifest_path

    report_path = out_dir / "participation_report.md"
    report = _render_report(
        f_base=float(f_base_result.f_base),
        calibration_loss=float(f_base_result.loss),
        subset_size=int(f_base_result.subset_size),
        hist_df=hist_df,
        null_df=null_df,
        corr_df=corr_df,
        corr_cat_df=corr_cat_df,
        cat_thresh_df=cat_threshold_df,
        out_paths=outputs,
    )
    report_path.write_text(report)
    outputs["report"] = report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integer participation + structural-noise coupling (Study 02)")
    parser.add_argument("--materials-csv", type=Path, default=Path("data/raw/materials_clusters_real_v7.csv"))
    parser.add_argument("--noise-csv", type=Path, help="Optional structural_noise_summary.csv for correlations")
    parser.add_argument("--materials", nargs="*", help="Optional list of materials to include")
    parser.add_argument("--subset", nargs="*", help="Names used to calibrate f_base (default=all)")
    parser.add_argument("--subset-category", nargs="*", help="Category filters for f_base calibration")
    parser.add_argument("--f-base-override", type=float, help="Skip calibration and force this f_base value")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/participation"))
    parser.add_argument("--max-n", type=int, default=500, help="Max integer multiplier explored during calibration grid")
    parser.add_argument("--grid-size", type=int, default=400, help="Number of coarse samples in f_base search")
    parser.add_argument("--refine-steps", type=int, default=200, help="Steps in the refinement window around the best coarse point")
    parser.add_argument("--bin-width", type=float, default=0.01, help="Bin width for |delta| histogram")
    parser.add_argument("--max-delta", type=float, default=0.5, help="Maximum |delta| to include in histogram")
    parser.add_argument("--thresholds", nargs="*", type=float, help="Thresholds for near-integer fractions")
    parser.add_argument("--permutations", type=int, default=500, help="Number of permutations for null models")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for permutations")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    main()
