"""Integer participation pipeline (v4.0): robust base-frequency calibration, noise correction, and null tests."""

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

DEFAULT_BOUNDS: Tuple[float, float] = (0.5, 5.0)
DEFAULT_PERMUTATIONS = 1000
DEFAULT_BOOTSTRAP = 500
LAMBDA_PENALTY = 1e-3
TOP_FRACTION = 0.2


@dataclass
class CalibrationResult:
    f_base: float
    cost: float
    median_delta: float
    mean_N: float
    n_samples: int


def _safe_series(values: Iterable[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[arr > 0]


def _search_grid(values: np.ndarray, bounds: Tuple[float, float], n: int = 400) -> np.ndarray:
    lo_bound, hi_bound = bounds
    lo = max(lo_bound, values.min() / 500.0, 1e-6)
    hi = min(hi_bound, max(values.max(), lo * 1.05))
    if hi <= lo:
        hi = min(hi_bound, lo * 1.05)
    return np.geomspace(lo, hi, num=n)


def calibrate_base_freq(values: np.ndarray, bounds: Tuple[float, float] = DEFAULT_BOUNDS, lambda_penalty: float = LAMBDA_PENALTY) -> CalibrationResult:
    """Robustly calibrate f_base using median |delta| + lambda*mean(N)."""

    vals = _safe_series(values)
    if vals.size == 0:
        raise ValueError("No finite/positive values provided for calibration.")
    lo_bound, hi_bound = bounds
    grid = _search_grid(vals, bounds)

    def _cost(f: float) -> Tuple[float, float]:
        if f < lo_bound or f > hi_bound:
            return float("inf"), float("inf")
        n = vals / f
        delta = np.abs(n - np.round(n))
        med_delta = float(np.median(delta))
        mean_n = float(np.mean(n))
        return med_delta + lambda_penalty * mean_n, med_delta

    costs = []
    for f in grid:
        total_cost, med_delta = _cost(f)
        costs.append((total_cost, med_delta, f))
    best_cost, best_med, best_f = min(costs, key=lambda x: x[0])
    best_f = float(np.clip(best_f, lo_bound, hi_bound))
    return CalibrationResult(f_base=float(best_f), cost=float(best_cost), median_delta=float(best_med), mean_N=float(np.mean(vals / best_f)), n_samples=int(vals.size))


def compute_participation(F_target: np.ndarray, f_base: float) -> Tuple[np.ndarray, np.ndarray]:
    n_vals = F_target / f_base
    delta = np.abs(n_vals - np.round(n_vals))
    return n_vals, delta


def zscore_by_family(noise: pd.Series, family: pd.Series) -> pd.Series:
    df = pd.concat([noise, family], axis=1)
    df.columns = ["noise", "family"]
    out = []
    for fam, group in df.groupby(df["family"].fillna("Unknown")):
        vals = group["noise"]
        mu = vals.mean()
        sigma = vals.std()
        if sigma == 0 or not math.isfinite(sigma):
            z = (vals - mu) * 0.0
        else:
            z = (vals - mu) / sigma
        out.append(z)
    return pd.concat(out).reindex(df.index)


def _median_delta(F_target: np.ndarray, f_base: float) -> float:
    _, delta = compute_participation(F_target, f_base)
    return float(np.median(delta[np.isfinite(delta)]))


def _fit_lognormal(values: np.ndarray) -> Tuple[float, float, float]:
    shape, loc, scale = stats.lognorm.fit(values, floc=0)
    return shape, loc, scale


def _fit_gamma(values: np.ndarray) -> Tuple[float, float, float]:
    a, loc, scale = stats.gamma.fit(values, floc=0)
    return a, loc, scale


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    ranks = []
    for x in a:
        ranks.append(np.sum(x > b) - np.sum(x < b))
    return float(np.sum(ranks) / (len(a) * len(b)))


def _spearman_with_ci(x: np.ndarray, y: np.ndarray, n_bootstrap: int = DEFAULT_BOOTSTRAP, seed: int = 123) -> Tuple[float, Tuple[float, float]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), (float("nan"), float("nan"))
    rho, _ = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        rb, _ = stats.spearmanr(x[idx], y[idx])
        if math.isfinite(rb):
            boot.append(rb)
    if not boot:
        return float(rho), (float("nan"), float("nan"))
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(rho), (float(low), float(high))


def _null_p_value(real_cost: float, null_costs: np.ndarray) -> float:
    if null_costs.size == 0:
        return float("nan")
    return float(np.mean(null_costs <= real_cost))


def run_integer_participation_pipeline(
    data_path: Path,
    noise_path: Path,
    mode: Sequence[str],
    hypotheses: Sequence[str],
    n_permutations: int,
    output_dir: Path,
    bounds: Tuple[float, float] = DEFAULT_BOUNDS,
    lambda_penalty: float = LAMBDA_PENALTY,
    seed: int = 123,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> None:
    print(f"[INFO] Loading materials from {data_path}")
    materials = pd.read_csv(data_path)
    noise = pd.read_csv(noise_path) if noise_path else pd.DataFrame()
    if "material" in noise.columns:
        noise = noise.rename(columns={"material": "name"})
    print(f"[INFO] Merging noise ({len(noise)} rows) into materials ({len(materials)} rows)")
    df = materials.merge(noise, on="name", how="left", suffixes=("", "_noise"))
    if "category" in df.columns:
        df["category"] = df["category"].fillna(df.get("category_noise")).fillna("Unknown")
    elif "category_noise" in df.columns:
        df["category"] = df["category_noise"].fillna("Unknown")
    else:
        df["category"] = "Unknown"
    df["predicted_noise"] = pd.to_numeric(df.get("predicted_noise"), errors="coerce")
    df["ThetaD_K"] = pd.to_numeric(df.get("ThetaD_K"), errors="coerce")
    df["Tc_K"] = pd.to_numeric(df.get("Tc_K"), errors="coerce")
    df["Tc_ideal"] = df["Tc_K"] * (1.0 + df["predicted_noise"].fillna(0.0))
    df["Fm_raw"] = df["ThetaD_K"] / df["Tc_K"]
    df["Fm_corr"] = df["ThetaD_K"] / df["Tc_ideal"]
    df["Fm_corr_div2"] = df["Fm_corr"] / 2.0
    print(f"[INFO] Valid rows for Fm_corr: {(df['Fm_corr'].notna()).sum()}")

    rng = np.random.default_rng(seed)
    results_rows: List[Dict[str, object]] = []
    metrics_by_hypothesis: Dict[str, Dict[str, float]] = {}

    def _calibrate_for_target(values: np.ndarray, fam_series: pd.Series, active_modes: Sequence[str]) -> Dict[str, float]:
        calib_map: Dict[str, float] = {}
        df_local = pd.DataFrame({"value": values, "family": fam_series})
        df_local = df_local[np.isfinite(df_local["value"])]
        if df_local.empty:
            return calib_map
        if "global" in active_modes:
            res = calibrate_base_freq(df_local["value"].to_numpy(dtype=float), bounds=bounds, lambda_penalty=lambda_penalty)
            calib_map["__global__"] = res.f_base
            print(f"[INFO] Global f_base={res.f_base:.6f} (median|δ|={res.median_delta:.4f}, n={res.n_samples})")
        if "per_family" in active_modes:
            df_local["family"] = df_local["family"].fillna("Unknown")
            for fam, group in df_local.groupby("family"):
                fam_vals = _safe_series(group["value"])
                if fam_vals.size == 0:
                    continue
                res = calibrate_base_freq(fam_vals, bounds=bounds, lambda_penalty=lambda_penalty)
                calib_map[fam] = res.f_base
                print(f"[INFO] f_base[{fam}]={res.f_base:.6f} (median|δ|={res.median_delta:.4f}, n={res.n_samples})")
        return calib_map

    for hyp in hypotheses:
        target_col = "Fm_corr" if hyp == "Fm" else "Fm_corr_div2"
        calib_mask = df[target_col].notna() & (df[target_col] <= 200)
        calib_vals = df.loc[calib_mask, target_col].to_numpy(dtype=float)
        calib_fams = df.loc[calib_mask, "category"]
        print(f"[INFO] Calibrating for hypothesis {hyp} with {calib_mask.sum()} samples")
        calib_map = _calibrate_for_target(calib_vals, calib_fams, mode)
        f_global = calib_map.get("__global__")
        df[f"f_base_{hyp}"] = df["category"].apply(lambda c: calib_map.get(c, f_global))
        df[f"N_{hyp}"] = df[target_col] / df[f"f_base_{hyp}"]
        df[f"delta_{hyp}"] = (df[f"N_{hyp}"] - np.round(df[f"N_{hyp}"])).abs()
        real_cost = float(np.median(df[f"delta_{hyp}"].to_numpy(dtype=float)))

        # Null models
        null_shuffle = []
        null_gamma = []
        null_lognorm = []
        valid_mask = df[target_col].notna() & df["Tc_K"].notna() & df[f"f_base_{hyp}"].notna()
        theta = df.loc[valid_mask, "ThetaD_K"].to_numpy(dtype=float)
        tc = df.loc[valid_mask, "Tc_ideal"].to_numpy(dtype=float)
        base_arr = df.loc[valid_mask, f"f_base_{hyp}"].to_numpy(dtype=float)
        for _ in range(max(1, n_permutations)):
            perm_theta = rng.permutation(theta)
            fm_perm = perm_theta / tc
            delta_perm = np.abs(fm_perm / base_arr - np.round(fm_perm / base_arr))
            null_shuffle.append(np.median(delta_perm))
        fm_sample = df[target_col].to_numpy(dtype=float)
        fm_sample = fm_sample[np.isfinite(fm_sample) & (fm_sample > 0)]
        if fm_sample.size:
            shape_l, loc_l, scale_l = _fit_lognormal(fm_sample)
            a_g, loc_g, scale_g = _fit_gamma(fm_sample)
            for _ in range(max(1, n_permutations)):
                sample_ln = stats.lognorm.rvs(shape_l, loc=loc_l, scale=scale_l, size=len(base_arr), random_state=rng)
                delta_ln = np.abs(sample_ln / base_arr - np.round(sample_ln / base_arr))
                null_lognorm.append(np.median(delta_ln))
                sample_g = stats.gamma.rvs(a_g, loc=loc_g, scale=scale_g, size=len(base_arr), random_state=rng)
                delta_g = np.abs(sample_g / base_arr - np.round(sample_g / base_arr))
                null_gamma.append(np.median(delta_g))
        metrics_by_hypothesis[hyp] = {
            "real_cost": real_cost,
            "p_shuffle": _null_p_value(real_cost, np.array(null_shuffle, dtype=float)),
            "p_lognorm": _null_p_value(real_cost, np.array(null_lognorm, dtype=float)),
            "p_gamma": _null_p_value(real_cost, np.array(null_gamma, dtype=float)),
        }
        print(f"[INFO] Hypothesis {hyp}: median|δ|={real_cost:.4f}, p_shuffle={metrics_by_hypothesis[hyp]['p_shuffle']:.4f}, "
              f"p_lognorm={metrics_by_hypothesis[hyp]['p_lognorm']:.4f}, p_gamma={metrics_by_hypothesis[hyp]['p_gamma']:.4f}")

    winner = min(metrics_by_hypothesis.items(), key=lambda x: x[1]["real_cost"])[0]
    win_stats = metrics_by_hypothesis[winner]
    print(f"[INFO] Winner hypothesis: {winner}")

    df["z_noise"] = zscore_by_family(df["predicted_noise"], df["category"])
    N_col = f"N_{winner}"
    delta_col = f"delta_{winner}"

    mask_valid = df[N_col].notna() & df[delta_col].notna() & (df[N_col] >= 1.0)
    rho, (ci_low, ci_high) = _spearman_with_ci(
        df.loc[mask_valid, delta_col].to_numpy(dtype=float),
        df.loc[mask_valid, "z_noise"].to_numpy(dtype=float),
        n_bootstrap=bootstrap,
        seed=seed,
    )

    quantile_cut = df[delta_col].quantile(TOP_FRACTION)
    near = df[df[delta_col] <= quantile_cut]["z_noise"].to_numpy(dtype=float)
    rest = df[df[delta_col] > quantile_cut]["z_noise"].to_numpy(dtype=float)
    cliffs = _cliffs_delta(near, rest)

    for row in df.itertuples():
        results_rows.append(
            {
                "name": row.name,
                "category": row.category,
                "hypothesis_winner": winner,
                "f_base_used": getattr(row, f"f_base_{winner}"),
                "Fm_raw": row.Fm_raw,
                "Fm_corr": row.Fm_corr,
                "N_value": getattr(row, N_col),
                "delta_value": getattr(row, delta_col),
                "predicted_noise": row.predicted_noise,
                "z_noise": getattr(row, "z_noise"),
                "p_shuffle": win_stats["p_shuffle"],
                "p_lognorm": win_stats["p_lognorm"],
                "p_gamma": win_stats["p_gamma"],
                "spearman_rho": rho,
                "spearman_ci_low": ci_low,
                "spearman_ci_high": ci_high,
                "cliffs_delta": cliffs,
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "participation_summary.csv"
    pd.DataFrame(results_rows).to_csv(summary_path, index=False)
    print(f"[INFO] Wrote summary to {summary_path}")

    manifest = {
        "data_path": str(data_path),
        "noise_path": str(noise_path),
        "mode": list(mode),
        "hypotheses": list(hypotheses),
        "n_permutations": n_permutations,
        "bounds": bounds,
        "lambda_penalty": lambda_penalty,
        "seed": seed,
        "bootstrap": bootstrap,
        "winner": winner,
        "stats": win_stats,
    }
    manifest_path = out_dir / "participation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[INFO] Wrote manifest to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute integer participation metrics with noise correction (v4.0).")
    parser.add_argument("--materials-csv", type=Path, default=Path("data/raw/materials_clusters_real_v7.csv"))
    parser.add_argument("--noise-csv", type=Path, default=Path("data/processed/run_w800_p7919-v7/structural_noise/structural_noise_summary.csv"))
    parser.add_argument("--mode", nargs="*", default=["global", "per_family"], choices=["global", "per_family"])
    parser.add_argument("--hypotheses", nargs="*", default=["Fm", "Fm_div_2"], choices=["Fm", "Fm_div_2"])
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/participation_v4"))
    parser.add_argument("--bounds", nargs=2, type=float, default=DEFAULT_BOUNDS, help="Bounds for f_base search (min max)")
    parser.add_argument("--lambda-penalty", type=float, default=LAMBDA_PENALTY, help="Penalty on mean N to avoid huge harmonics")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_integer_participation_pipeline(
        data_path=args.materials_csv,
        noise_path=args.noise_csv,
        mode=args.mode,
        hypotheses=args.hypotheses,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        bounds=tuple(args.bounds),
        lambda_penalty=args.lambda_penalty,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )


if __name__ == "__main__":
    main()
