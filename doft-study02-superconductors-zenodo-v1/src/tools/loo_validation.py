"""Leave-one-out validation by family for DOFT structural noise outputs."""

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

from src.compute_structural_noise import PRIMES


def _flatten_entry(entry: dict, subnets: List[str]) -> Dict[str, float]:
    flat: Dict[str, float] = {"xi": float(entry.get("xi", 0.0) or 0.0)}
    for param_key in ("delta_T", "delta_space", "delta_P"):
        acc = {str(p): [] for p in PRIMES}
        block = entry.get(param_key, {})
        if isinstance(block, (int, float)):
            for prime in PRIMES:
                acc[str(prime)].append(float(block))
        elif isinstance(block, dict):
            for subnet in subnets:
                vec = block.get(subnet)
                if isinstance(vec, (int, float)):
                    for prime in PRIMES:
                        acc[str(prime)].append(float(vec))
                elif isinstance(vec, dict):
                    for prime in PRIMES:
                        val = vec.get(str(prime))
                        if isinstance(val, (int, float)):
                            acc[str(prime)].append(float(val))
        for prime in PRIMES:
            flat[f"{param_key}_{prime}"] = float(np.mean(acc[str(prime)])) if acc[str(prime)] else 0.0
    return flat


def _mean_template(entries: Dict[str, dict], subnets: List[str]) -> Dict[str, float]:
    if not entries:
        return {}
    stacked = [_flatten_entry(entry, subnets) for entry in entries.values()]
    keys = stacked[0].keys()
    template: Dict[str, float] = {}
    for key in keys:
        vals = [float(sample.get(key, 0.0)) for sample in stacked]
        template[key] = float(np.mean(vals)) if vals else 0.0
    return template


def _mae(target: Dict[str, float], predicted: Dict[str, float], prefix: str) -> float:
    vals = []
    for key, val in target.items():
        if not key.startswith(prefix):
            continue
        vals.append(abs(val - predicted.get(key, 0.0)))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def loo_validation(noise_json: Path) -> pd.DataFrame:
    data = json.loads(noise_json.read_text())
    fam_map: Dict[str, List[str]] = {}
    subnets_by_mat: Dict[str, List[str]] = {}
    for name, entry in data.items():
        fam = str(entry.get("category", "Unknown"))
        fam_map.setdefault(fam, []).append(name)
        subnets_by_mat[name] = sorted(entry.get("subnets", list(entry.get("delta_T", {}).keys())))

    rows = []
    for fam, materials in fam_map.items():
        if len(materials) < 3:
            continue
        for left_out in materials:
            train_entries = {m: data[m] for m in materials if m != left_out}
            template = _mean_template(train_entries, subnets_by_mat.get(left_out, []))
            target = _flatten_entry(data[left_out], subnets_by_mat.get(left_out, []))
            mae_xi = abs(target["xi"] - template.get("xi", 0.0))
            mae_delta_T = _mae(target, template, "delta_T_")
            mae_delta_space = _mae(target, template, "delta_space_")
            mae_delta_P = _mae(target, template, "delta_P_")
            all_vals = [
                abs(target.get(k, 0.0) - template.get(k, 0.0))
                for k in target.keys()
                if k == "xi" or k.startswith("delta_")
            ]
            rows.append(
                {
                    "family": fam,
                    "left_out": left_out,
                    "mae_xi": mae_xi,
                    "mae_delta_T": mae_delta_T,
                    "mae_delta_space": mae_delta_space,
                    "mae_delta_P": mae_delta_P,
                    "mae_all": float(np.mean(all_vals)) if all_vals else 0.0,
                    "mse_all": float(np.mean([v * v for v in all_vals])) if all_vals else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-out validation over families.")
    parser.add_argument("--noise-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="CSV to store LOO results")
    args = parser.parse_args()
    df = loo_validation(args.noise_json)
    if args.output:
        df.to_csv(args.output, index=False)
    else:
        print(df)


if __name__ == "__main__":
    main()
