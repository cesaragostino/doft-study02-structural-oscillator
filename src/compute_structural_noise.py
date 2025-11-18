"""Compute structural noise factors from Study 01 (v6) materials.

The script ingests the multiband materials CSV (e.g. data/raw/materials_clusters_real_v6.csv),
computes DOFT ratios per subred (th->Δ, Δ->ΘD, ΘD->EF), finds the closest integer locks
on the 2-3-5-7 grid, and aggregates mismatch/M_struct metrics across subredes.

Usage example:

    python3 scripts/compute_structural_noise.py \
      --materials-csv data/raw/materials_clusters_real_v6.csv \
      --output-csv outputs/structural_noise_summary.csv \
      --output-json outputs/structural_noise_values.json \
      --fit-by-category
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PRIMES: Tuple[int, int, int, int] = (2, 3, 5, 7)
MEV_TO_K = 11.6045
EV_TO_K = 11604.5
RATIO_KEYS: Tuple[str, str, str] = ("th_to_delta", "delta_to_theta", "theta_to_ef")
PRIME_STR: Tuple[str, str, str, str] = tuple(str(p) for p in PRIMES)


# ---- primitive helpers -----------------------------------------------------


def mev_to_k(value: float) -> float:
    return value * MEV_TO_K


def ev_to_k(value: float) -> float:
    return value * EV_TO_K


def is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def prime_product_from_exponents(exps: Sequence[int]) -> float:
    value = 1.0
    for exp, prime in zip(exps, PRIMES):
        value *= prime ** exp
    return value


def best_integer_lock(value: float, min_exp: int = -6, max_exp: int = 6) -> Tuple[float, Tuple[int, int, int, int]]:
    """Return the closest integer lock on the 2-3-5-7 grid to ``value``.

    The search exhausts exponent combinations in [min_exp, max_exp] (inclusive)
    and minimises |log(candidate) - log(value)|.
    """

    if value <= 0 or not math.isfinite(value):
        return float("nan"), (0, 0, 0, 0)

    log_target = math.log(value)
    best_diff = float("inf")
    best_exps = (0, 0, 0, 0)
    for a2 in range(min_exp, max_exp + 1):
        for a3 in range(min_exp, max_exp + 1):
            for a5 in range(min_exp, max_exp + 1):
                for a7 in range(min_exp, max_exp + 1):
                    candidate = prime_product_from_exponents((a2, a3, a5, a7))
                    diff = abs(math.log(candidate) - log_target)
                    if diff < best_diff:
                        best_diff = diff
                        best_exps = (a2, a3, a5, a7)
    lock_value = prime_product_from_exponents(best_exps)
    return lock_value, best_exps


# ---- domain helpers --------------------------------------------------------


@dataclass
class SubnetRatios:
    ratios: Dict[str, float]
    locks: Dict[str, Tuple[int, int, int, int]]


def compute_ratios_for_row(row: pd.Series) -> SubnetRatios:
    tc = row.get("Tc_K")
    gap_mev = row.get("Gap_meV")
    theta_d = row.get("ThetaD_K")
    ef_ev = row.get("EF_eV")

    ratios: Dict[str, float] = {}
    locks: Dict[str, Tuple[int, int, int, int]] = {}

    gap_k = mev_to_k(gap_mev) if is_finite(gap_mev) else None
    ef_k = ev_to_k(ef_ev) if is_finite(ef_ev) else None

    if is_finite(tc) and gap_k and gap_k > 0 and tc > 0:
        ratios["th_to_delta"] = gap_k / tc
    if gap_k and is_finite(theta_d) and theta_d > 0 and gap_k > 0:
        ratios["delta_to_theta"] = theta_d / gap_k
    if is_finite(theta_d) and ef_k and ef_k > 0 and theta_d > 0:
        ratios["theta_to_ef"] = ef_k / theta_d

    for key, value in ratios.items():
        _, exps = best_integer_lock(value)
        locks[key] = exps

    return SubnetRatios(ratios=ratios, locks=locks)


def mismatch_between(exps_a: Tuple[int, int, int, int], exps_b: Tuple[int, int, int, int]) -> int:
    return int(sum(abs(a - b) for a, b in zip(exps_a, exps_b)))


def compute_pair_metrics(
    locks_by_ratio_a: Dict[str, Tuple[int, int, int, int]],
    locks_by_ratio_b: Dict[str, Tuple[int, int, int, int]],
    ratios_a: Dict[str, float],
    ratios_b: Dict[str, float],
) -> Tuple[float, float, int, List[float], List[int]]:
    mismatch_total = 0.0
    mstruct = 0.0
    used_terms = 0
    exp_diffs_sum = [0.0, 0.0, 0.0, 0.0]
    exp_diffs_count = [0, 0, 0, 0]
    for key in RATIO_KEYS:
        exps_a = locks_by_ratio_a.get(key)
        exps_b = locks_by_ratio_b.get(key)
        if exps_a is not None and exps_b is not None:
            mismatch_total += mismatch_between(exps_a, exps_b)
            for idx, (ea, eb) in enumerate(zip(exps_a, exps_b)):
                exp_diffs_sum[idx] += abs(ea - eb)
                exp_diffs_count[idx] += 1

        v1 = ratios_a.get(key)
        v2 = ratios_b.get(key)
        if v1 is None or v2 is None or v1 <= 0 or v2 <= 0:
            continue
        l_eff = math.sqrt(v1 * v2)
        delta = math.log(v1 / l_eff)
        mstruct += 2.0 * (delta * delta)
        used_terms += 1

    return mismatch_total, mstruct, used_terms, exp_diffs_sum, exp_diffs_count


# ---- main pipeline ---------------------------------------------------------


def calc_zeta(x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    if x.size == 0 or np.allclose(x, 0):
        return 0.0, None
    zeta = float(np.dot(x, y) / np.dot(x, x))
    corr = None
    if x.size >= 2:
        try:
            corr = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            corr = None
    return zeta, corr


def summarize_material(name: str, group: pd.DataFrame) -> Dict[str, object]:
    subnets = sorted(set(group["sub_network"]))
    subnet_entries: Dict[str, SubnetRatios] = {}
    for subnet, rows in group.groupby("sub_network"):
        subnet_entries[subnet] = compute_ratios_for_row(rows.iloc[0])

    pair_metrics = []
    exp_diff_sum = [0.0, 0.0, 0.0, 0.0]
    exp_diff_count = [0, 0, 0, 0]
    for a, b in combinations(subnets, 2):
        entry_a = subnet_entries[a]
        entry_b = subnet_entries[b]
        mismatch_total, mstruct, used_terms, diffs_sum, diffs_count = compute_pair_metrics(
            entry_a.locks, entry_b.locks, entry_a.ratios, entry_b.ratios
        )
        if used_terms == 0:
            continue
        pair_metrics.append((mismatch_total, mstruct))
        exp_diff_sum = [sa + sb for sa, sb in zip(exp_diff_sum, diffs_sum)]
        exp_diff_count = [ca + cb for ca, cb in zip(exp_diff_count, diffs_count)]

    if not pair_metrics:
        return {
            "name": name,
            "subnets": subnets,
            "num_subnets": len(subnets),
            "pair_count": 0,
            "mismatch_sum": 0.0,
            "mismatch_mean": 0.0,
            "N_mismatch": 0.0,
            "M_struct_sum": 0.0,
            "M_struct_mean": 0.0,
            "ratios_used_mean": 0.0,
            "exp_diff_mean_2": 0.0,
            "exp_diff_mean_3": 0.0,
            "exp_diff_mean_5": 0.0,
            "exp_diff_mean_7": 0.0,
        }

    mismatch_sum = sum(m for m, _ in pair_metrics)
    mstruct_sum = sum(ms for _, ms in pair_metrics)
    pair_count = len(pair_metrics)
    mismatch_mean = mismatch_sum / pair_count
    mstruct_mean = mstruct_sum / pair_count
    # Each term corresponds to one ratio used; divide by pair_count to get mean ratios used per pair.
    ratios_used_mean = 0.0
    if pair_count > 0:
        total_terms = sum(
            sum(1 for key in RATIO_KEYS if key in subnet_entries[a].ratios and key in subnet_entries[b].ratios)
            for a, b in combinations(subnets, 2)
        )
        ratios_used_mean = total_terms / pair_count

    exp_diff_mean = [
        (exp_diff_sum[idx] / exp_diff_count[idx]) if exp_diff_count[idx] > 0 else 0.0 for idx in range(4)
    ]

    return {
        "name": name,
        "subnets": subnets,
        "num_subnets": len(subnets),
        "pair_count": pair_count,
        "mismatch_sum": mismatch_sum,
        "mismatch_mean": mismatch_mean,
        "N_mismatch": mismatch_mean,
        "M_struct_sum": mstruct_sum,
        "M_struct_mean": mstruct_mean,
        "ratios_used_mean": ratios_used_mean,
        "exp_diff_mean_2": exp_diff_mean[0],
        "exp_diff_mean_3": exp_diff_mean[1],
        "exp_diff_mean_5": exp_diff_mean[2],
        "exp_diff_mean_7": exp_diff_mean[3],
    }


def load_existing_mstruct(path: Optional[Path]) -> Dict[str, float]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    mapping: Dict[str, float] = {}
    if "name" in df.columns:
        for row in df.itertuples():
            if hasattr(row, "M_struct"):
                mapping[str(row.name)] = float(getattr(row, "M_struct"))
    return mapping


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.materials_csv)
    if args.materials:
        allowed = set(args.materials)
        df = df[df["name"].isin(allowed)]

    # Only keep materials with 2+ subnets
    multi = df.groupby("name").filter(lambda g: g["sub_network"].nunique() >= 2)

    existing_mstruct = load_existing_mstruct(Path(args.use_existing_mstruct) if args.use_existing_mstruct else None)

    summaries = []
    for name, group in multi.groupby("name"):
        summary = summarize_material(name, group)
        category = str(group["category"].iloc[0]) if "category" in group.columns else ""
        summary["category"] = category
        # Override M_struct if provided
        if name in existing_mstruct:
            summary["M_struct_mean"] = existing_mstruct[name]
        summaries.append(summary)

    if not summaries:
        print("No multiband materials found with the given filters.")
        return

    summary_df = pd.DataFrame(summaries)

    # Calibration of zeta
    zeta_global, corr_global = calc_zeta(
        summary_df["mismatch_mean"].to_numpy(dtype=float),
        summary_df["M_struct_mean"].to_numpy(dtype=float),
    )

    # Calibration of k_exp per prime using exp_diff_mean_* vs M_struct_mean
    k_exp: Dict[str, float] = {str(p): 0.0 for p in PRIMES}
    corr_exp: Dict[str, Optional[float]] = {str(p): None for p in PRIMES}
    x_exp = summary_df[[f"exp_diff_mean_{p}" for p in PRIMES]].to_numpy(dtype=float)
    y = summary_df["M_struct_mean"].to_numpy(dtype=float)
    for idx, prime in enumerate(PRIMES):
        k_val, c_val = calc_zeta(x_exp[:, idx], y)
        k_exp[str(prime)] = k_val
        corr_exp[str(prime)] = c_val

    # Category-specific zeta if requested
    zeta_by_cat: Dict[str, float] = {}
    corr_by_cat: Dict[str, Optional[float]] = {}
    if args.fit_by_category and "category" in summary_df.columns:
        for cat, cat_df in summary_df.groupby("category"):
            z, c = calc_zeta(
                cat_df["mismatch_mean"].to_numpy(dtype=float),
                cat_df["M_struct_mean"].to_numpy(dtype=float),
            )
            zeta_by_cat[cat] = z
            corr_by_cat[cat] = c

    def select_zeta(category: str) -> float:
        if args.fit_by_category and category in zeta_by_cat and zeta_by_cat[category] != 0.0:
            return zeta_by_cat[category]
        return zeta_global

    summary_df["zeta_used"] = summary_df["category"].apply(select_zeta)
    summary_df["predicted_noise"] = summary_df["zeta_used"] * summary_df["mismatch_mean"]
    # xi_exp per prime
    for prime in PRIMES:
        col_name = f"xi_exp_{prime}"
        summary_df[col_name] = summary_df[f"exp_diff_mean_{prime}"] * k_exp[str(prime)]
    summary_df["k_skin"] = args.k_skin

    # Serialization
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    if args.output_json:
        mapping = {}
        for row in summary_df.itertuples():
            xi_exp = {str(p): getattr(row, f"xi_exp_{p}") for p in PRIMES}
            mapping[row.name] = {
                "xi": row.predicted_noise,
                "xi_exp": xi_exp,
                "k_skin": args.k_skin,
            }
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(mapping, indent=2))

    print(f"ζ (global) = {zeta_global:.6f} | corr={corr_global}")
    if args.fit_by_category:
        for cat, z in zeta_by_cat.items():
            print(f"  {cat}: ζ={z:.6f} | corr={corr_by_cat.get(cat)}")
    print(f"k_exp per prime: {k_exp} | corr_exp={corr_exp}")
    print(f"Wrote summary to {output_csv}")
    if args.output_json:
        print(f"Wrote predicted noise JSON to {args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute structural noise factors from Study 01 v6 data")
    parser.add_argument("--materials-csv", type=Path, default=Path("data/raw/materials_clusters_real_v6.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/structural_noise_summary.csv"))
    parser.add_argument("--output-json", type=Path, help="Optional path to write {material: xi} mapping")
    parser.add_argument("--materials", nargs="*", help="Optional list of materials to include")
    parser.add_argument("--fit-by-category", action="store_true", help="Calibrate ζ separately per category")
    parser.add_argument(
        "--use-existing-mstruct",
        type=Path,
        help="Optional CSV with precomputed M_struct (columns: name, M_struct)",
    )
    parser.add_argument("--k-skin", type=float, default=0.05, help="Skin coupling coefficient for structural noise term")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
