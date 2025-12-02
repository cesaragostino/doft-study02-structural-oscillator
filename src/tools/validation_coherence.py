"""Validate integer participation against experimental coherence lengths.

Matches experimental coherence-length entries to model outputs, runs a log-log
regression N = A * xi0**alpha, and produces a scatter + fitted trend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _match_name(pattern: str, candidates: Iterable[str]) -> Optional[str]:
    """Return the best candidate that contains pattern (case-insensitive)."""

    pat = pattern.lower().strip()
    best = None
    best_score = -1
    for name in candidates:
        lname = name.lower()
        if pat in lname:
            # Prefer closer length to pattern and longer match score
            score = -abs(len(lname) - len(pat))
            if score > best_score:
                best_score = score
                best = name
    return best


def _load_participation(part_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(part_csv)
    df["name"] = df["name"].astype(str)
    df["N_value"] = pd.to_numeric(df.get("N_value"), errors="coerce")
    df["category"] = df.get("category", "Unknown").fillna("Unknown").astype(str)
    grouped = (
        df.groupby("name")
        .agg(N_mean=("N_value", "mean"), category=("category", "first"))
        .reset_index()
    )
    return grouped


def build_validation_table(part_csv: Path, exp_csv: Path) -> pd.DataFrame:
    part = _load_participation(part_csv)
    exp = pd.read_csv(exp_csv)
    exp["material_pattern"] = exp["material_pattern"].astype(str)
    exp["xi0_nm"] = pd.to_numeric(exp["xi0_nm"], errors="coerce")
    exp["source_ref"] = exp["source_ref"].astype(str)
    rows: List[Dict[str, object]] = []
    for row in exp.itertuples():
        match = _match_name(row.material_pattern, part["name"])
        if match is None:
            continue
        entry = part[part["name"] == match].iloc[0]
        rows.append(
            {
                "Material": match,
                "Family": entry["category"],
                "N_model": float(entry["N_mean"]),
                "Xi0_exp": float(row.xi0_nm),
                "Reference": row.source_ref,
                "Pattern": row.material_pattern,
            }
        )
    return pd.DataFrame(rows)


def run_regression(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    x = df["Xi0_exp"].to_numpy(dtype=float)
    y = df["N_model"].to_numpy(dtype=float)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    lx = np.log(x)
    ly = np.log(y)
    res = stats.linregress(lx, ly)
    alpha = res.slope
    logA = res.intercept
    r2 = res.rvalue ** 2
    pval = res.pvalue
    return alpha, logA, r2, pval


def plot_scatter(df: pd.DataFrame, alpha: float, logA: float, r2: float, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    families = df["Family"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
    for col, fam in zip(colors, families):
        mask = df["Family"] == fam
        plt.scatter(df.loc[mask, "Xi0_exp"], df.loc[mask, "N_model"], color=col, alpha=0.7, label=fam)
        for _, row in df.loc[mask].iterrows():
            plt.text(row["Xi0_exp"] * 1.02, row["N_model"] * 1.02, row["Material"], fontsize=7, color=col)
    if not (np.isnan(alpha) or np.isnan(logA)):
        x_line = np.linspace(df["Xi0_exp"].min() * 0.9, df["Xi0_exp"].max() * 1.1, 100)
        y_line = np.exp(logA) * x_line ** alpha
        plt.plot(x_line, y_line, "k--", label=f"N ~ xi0^{alpha:.2f}, R^2={r2:.3f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Coherence length $\xi_0$ (nm)")
    plt.ylabel(r"Participation number $N$")
    plt.title("Experimental coherence length vs model participation")
    plt.legend(fontsize="x-small", ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate participation numbers against experimental coherence lengths.")
    parser.add_argument("--participation-csv", type=Path, required=True, help="Path to participation_summary.csv")
    parser.add_argument("--experimental-csv", type=Path, default=Path("data/raw/experimental_coherence.csv"), help="Experimental coherence lengths CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/participation_validation"), help="Output directory for stats and plots")
    parser.add_argument("--seed", type=int, default=123, help="Seed placeholder (for future randomness; currently unused)")
    args = parser.parse_args()

    df = build_validation_table(args.participation_csv, args.experimental_csv)
    if df.empty:
        print("[WARN] No matches between experimental patterns and participation names.")
        return
    alpha, logA, r2, pval = run_regression(df)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "validation_coherence_stats.csv", index=False)
    plot_scatter(df, alpha, logA, r2, out_dir / "validation_coherence_N.png")
    print(f"[INFO] Matches: {len(df)}")
    print(f"[INFO] Fit: N ~ xi0^{alpha:.3f}, R^2={r2:.3f}, p={pval:.3e}")
    print(f"[INFO] Wrote stats to {out_dir / 'validation_coherence_stats.csv'}")
    print(f"[INFO] Wrote figure to {out_dir / 'validation_coherence_N.png'}")


if __name__ == "__main__":
    main()
