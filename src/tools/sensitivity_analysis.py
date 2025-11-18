"""Sensitivity analysis utilities for DOFT parameters.

Perturbs material inputs and measures stability of calibrated parameters
(xi, delta_T, delta_space, delta_P) by recomputing mismatch proxies.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.compute_structural_noise import SubnetRatios, compute_pair_metrics, compute_ratios_for_row, mismatch_between, summarize_material


def _extract_baseline(noise_json: Path, material: str) -> dict:
    data = json.loads(noise_json.read_text())
    entry = data.get(material, {})
    return {
        "xi": entry.get("xi"),
        "delta_T": entry.get("delta_T", {}),
        "delta_space": entry.get("delta_space", {}),
        "delta_P": entry.get("delta_P", {}),
    }


def _summarize_material_from_rows(rows: pd.DataFrame) -> dict:
    return summarize_material(str(rows["name"].iloc[0]), rows)


def _perturb_rows(group: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    perturbed = group.copy()
    for col in ("Tc_K", "Gap_meV", "ThetaD_K", "EF_eV", "pressure_GPa"):
        if col in perturbed.columns:
            noise = np.random.uniform(-epsilon, epsilon, size=len(perturbed))
            perturbed[col] = perturbed[col] * (1.0 + noise)
    return perturbed


def sensitivity_analysis(material: str, materials_csv: Path, noise_json: Path, n_perturbations: int = 100, epsilon: float = 0.05) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(materials_csv)
    df = df[df["name"] == material]
    if df.empty:
        raise ValueError(f"Material {material} not found in {materials_csv}")
    base_summary = _summarize_material_from_rows(df)
    baseline = _extract_baseline(noise_json, material)
    ratios = []
    p_deltas: Dict[str, List[float]] = {"xi": []}

    def _flatten(vec: dict) -> List[float]:
        arr: List[float] = []
        for subnet_vec in vec.values():
            if isinstance(subnet_vec, dict):
                arr.extend([float(v) for v in subnet_vec.values()])
        return arr

    base_xi = float(baseline.get("xi") or 0.0)
    base_delta_values = {
        "delta_T": np.array(_flatten(baseline.get("delta_T", {})) or [0.0]),
        "delta_space": np.array(_flatten(baseline.get("delta_space", {})) or [0.0]),
        "delta_P": np.array(_flatten(baseline.get("delta_P", {})) or [0.0]),
    }

    for _ in range(max(1, n_perturbations)):
        perturbed = _perturb_rows(df, epsilon)
        summary = _summarize_material_from_rows(perturbed)
        # Proxy xi: scale baseline xi by mismatch_ratio if available
        xi_val = base_xi
        if base_summary.get("mismatch_mean", 0) not in (0, None):
            ratio = (summary.get("mismatch_mean", 0) or 0.0) / max(base_summary.get("mismatch_mean", 1e-6), 1e-6)
            xi_val = base_xi * ratio
        p_deltas["xi"].append(xi_val)
        for key in ("delta_T", "delta_space", "delta_P"):
            base_vec = base_delta_values[key]
            scale = (summary.get("mismatch_mean", 0) or 0.0) / max(base_summary.get("mismatch_mean", 1e-6), 1e-6)
            perturbed_vec = base_vec * scale
            p_deltas.setdefault(key, []).append(np.mean(perturbed_vec))

    stats: Dict[str, Tuple[float, float]] = {}
    for key, values in p_deltas.items():
        arr = np.array(values, dtype=float)
        stats[key] = (float(np.mean(arr)), float(np.std(arr)))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity analysis for a material.")
    parser.add_argument("--materials-csv", type=Path, required=True)
    parser.add_argument("--noise-json", type=Path, required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--n-perturbations", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="Optional CSV to write stats")
    args = parser.parse_args()

    stats = sensitivity_analysis(args.material, args.materials_csv, args.noise_json, args.n_perturbations, args.epsilon)
    if args.output:
        rows = [{"parameter": k, "mean": v[0], "std": v[1]} for k, v in stats.items()]
        pd.DataFrame(rows).to_csv(args.output, index=False)
    else:
        print(stats)


if __name__ == "__main__":
    main()
