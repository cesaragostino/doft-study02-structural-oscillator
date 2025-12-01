"""Sensitivity analysis utilities for DOFT parameters.

Perturbs material inputs (Tc, Δ, ΘD, EF, pressure) and measures stability of
calibrated parameters (xi, delta_T, delta_space, delta_P). Returns mean, std,
and 95% confidence intervals per parameter so unstable fits can be flagged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compute_structural_noise import (
    DEFAULT_LAMBDA_GEO,
    DEFAULT_LAMBDA_PRESSURE_GEO,
    PRIMES,
    initialise_delta_vectors,
    initialise_pressure_vectors,
    summarize_material,
)


def _aggregate_by_prime(delta_block: object, subnets: List[str]) -> Dict[str, float]:
    """Average values across subnets for each prime."""

    acc: Dict[str, List[float]] = {str(p): [] for p in PRIMES}
    if isinstance(delta_block, (int, float)):
        for prime in PRIMES:
            acc[str(prime)].append(float(delta_block))
    elif isinstance(delta_block, dict):
        for subnet in subnets:
            vec = delta_block.get(subnet)
            if isinstance(vec, (int, float)):
                for prime in PRIMES:
                    acc[str(prime)].append(float(vec))
            elif isinstance(vec, dict):
                for prime in PRIMES:
                    key = str(prime)
                    val = vec.get(key)
                    if isinstance(val, (int, float)):
                        acc[key].append(float(val))
    return {key: (float(np.mean(vals)) if vals else 0.0) for key, vals in acc.items()}


def _perturb_numeric_columns(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    perturbed = df.copy()
    for col in ("Tc_K", "Gap_meV", "ThetaD_K", "EF_eV", "pressure_GPa"):
        if col in perturbed.columns:
            noise = np.random.uniform(-epsilon, epsilon, size=len(perturbed))
            perturbed[col] = perturbed[col] * (1.0 + noise)
    return perturbed


def _load_baseline(noise_json: Path, material: str) -> dict:
    data = json.loads(noise_json.read_text())
    entry = data.get(material, {})
    return entry


def sensitivity_analysis(
    material: str,
    materials_csv: Path,
    noise_json: Path,
    n_perturbations: int = 100,
    epsilon: float = 0.05,
    c_pressure: float = 0.01,
    pressure_ref: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Run perturbation-based sensitivity for a single material."""

    df = pd.read_csv(materials_csv)
    if "pressure_GPa" not in df.columns:
        raise ValueError("materials CSV must contain a 'pressure_GPa' column; no inference is applied.")
    df["pressure_GPa"] = pd.to_numeric(df["pressure_GPa"], errors="coerce")
    if df["pressure_GPa"].isna().any():
        names = df[df["pressure_GPa"].isna()]["name"].astype(str).unique().tolist()
        raise ValueError(f"'pressure_GPa' has non-numeric/missing values for: {names[:10]} (showing up to 10).")
    df = df[df["name"] == material]
    if df.empty:
        raise ValueError(f"Material {material} not found in {materials_csv}")
    subnets = sorted(df["sub_network"].dropna().astype(str).unique().tolist())
    category = str(df["category"].iloc[0]) if "category" in df.columns else "Unknown"
    baseline_summary = summarize_material(material, df)
    baseline_mismatch = float(baseline_summary.get("mismatch_mean", 0.0) or 0.0)

    baseline_entry = _load_baseline(noise_json, material)
    baseline_xi = float(baseline_entry.get("xi", 0.0) or 0.0)
    lambda_geo_vec = tuple(
        baseline_entry.get("lambda_geo", {}).get(str(p), DEFAULT_LAMBDA_GEO[idx]) for idx, p in enumerate(PRIMES)
    )
    lambda_pressure_geo_vec = tuple(
        baseline_entry.get("lambda_pressure_geo", {}).get(str(p), DEFAULT_LAMBDA_PRESSURE_GEO[idx])
        for idx, p in enumerate(PRIMES)
    )
    lambda_band_map = baseline_entry.get("lambda_band", {}) or {subnet: 1.0 for subnet in subnets}
    lambda_pressure_band_map = baseline_entry.get("lambda_pressure_band", {}) or {
        subnet: lambda_band_map.get(subnet, 1.0) for subnet in subnets
    }
    baseline_delta_T = _aggregate_by_prime(baseline_entry.get("delta_T", {}), subnets)
    baseline_delta_space = _aggregate_by_prime(baseline_entry.get("delta_space", {}), subnets)
    baseline_delta_P = _aggregate_by_prime(baseline_entry.get("delta_P", {}), subnets)

    baseline_values = {"xi": baseline_xi}
    for prime in PRIMES:
        key = str(prime)
        baseline_values[f"delta_T_{key}"] = baseline_delta_T.get(key, 0.0)
        baseline_values[f"delta_space_{key}"] = baseline_delta_space.get(key, 0.0)
        baseline_values[f"delta_P_{key}"] = baseline_delta_P.get(key, 0.0)

    samples: Dict[str, List[float]] = {key: [] for key in baseline_values.keys()}

    for _ in range(max(1, n_perturbations)):
        perturbed_rows = _perturb_numeric_columns(df, epsilon)
        summary = summarize_material(material, perturbed_rows)
        exp_diff_mean = [
            summary.get(f"exp_diff_mean_{p}", 0.0) or 0.0 for p in PRIMES
        ]
        mismatch_mean = float(summary.get("mismatch_mean", 0.0) or 0.0)
        subnet_delta_T, subnet_delta_space, lambda_band_map_pert = initialise_delta_vectors(
            subnets,
            exp_diff_mean,
            mismatch_mean,
            lambda_geo_vec,
            default_delta_T=0.0,
            default_delta_space=0.0,
            category=category,
        )
        # Prefer material-specific lambda_pressure_band, otherwise reuse inferred lambda_band
        lambda_pressure_band = {subnet: lambda_pressure_band_map.get(subnet, lambda_band_map_pert.get(subnet, 1.0)) for subnet in subnets}
        subnet_delta_P = initialise_pressure_vectors(
            subnets,
            float(summary.get("pressure_GPa", 0.0) or 0.0),
            lambda_pressure_geo_vec,
            lambda_pressure_band,
            c_pressure=c_pressure,
            pressure_ref=pressure_ref,
        )

        xi_scaled = baseline_xi
        if baseline_mismatch > 0 and mismatch_mean > 0:
            xi_scaled = baseline_xi * (mismatch_mean / baseline_mismatch)

        agg_delta_T = _aggregate_by_prime(subnet_delta_T, subnets)
        agg_delta_space = _aggregate_by_prime(subnet_delta_space, subnets)
        agg_delta_P = _aggregate_by_prime(subnet_delta_P, subnets)

        samples["xi"].append(xi_scaled)
        for prime in PRIMES:
            key = str(prime)
            samples[f"delta_T_{key}"].append(agg_delta_T.get(key, 0.0))
            samples[f"delta_space_{key}"].append(agg_delta_space.get(key, 0.0))
            samples[f"delta_P_{key}"].append(agg_delta_P.get(key, 0.0))

    stats: Dict[str, Dict[str, float]] = {}
    for key, values in samples.items():
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        ci_range = 1.96 * std
        stats[key] = {
            "baseline": baseline_values.get(key, 0.0),
            "mean": mean,
            "std": std,
            "ci95_low": float(mean - ci_range),
            "ci95_high": float(mean + ci_range),
            "rel_std": float(std / (abs(baseline_values.get(key, 1e-6)) or 1e-6)),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity analysis for a material.")
    parser.add_argument("--materials-csv", type=Path, required=True)
    parser.add_argument("--noise-json", type=Path, required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--n-perturbations", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--c-pressure", type=float, default=0.01)
    parser.add_argument("--pressure-ref", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Optional CSV to write stats")
    args = parser.parse_args()

    stats = sensitivity_analysis(
        args.material,
        args.materials_csv,
        args.noise_json,
        n_perturbations=args.n_perturbations,
        epsilon=args.epsilon,
        c_pressure=args.c_pressure,
        pressure_ref=args.pressure_ref,
    )
    if args.output:
        rows = []
        for key, entry in stats.items():
            row = {"parameter": key}
            row.update(entry)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
    else:
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
