"""Compute simple AIC/BIC proxies for DOFT experiments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def compute_metrics(simulator_summary: Path, model_label: str, k_params: int) -> pd.DataFrame:
    df = pd.read_csv(simulator_summary)
    if "total_loss" not in df.columns or df.empty:
        raise ValueError("simulator_summary.csv missing or has no data")
    n = len(df)
    # Use total_loss as proxy for -2 ln(L)
    loss_sum = df["total_loss"].sum()
    aic = 2 * k_params + loss_sum
    bic = k_params * math.log(max(n, 1)) + loss_sum
    return pd.DataFrame(
        [
            {
                "model": model_label,
                "k_params": k_params,
                "n_samples": n,
                "total_loss": loss_sum,
                "AIC_proxy": aic,
                "BIC_proxy": bic,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AIC/BIC proxies for a DOFT run.")
    parser.add_argument("--simulator-summary", type=Path, required=True)
    parser.add_argument("--model-label", default="vector_pressure")
    parser.add_argument("--k-params", type=int, default=21, help="Parameter count proxy for the model")
    parser.add_argument("--output", type=Path, help="CSV to write the metrics")
    args = parser.parse_args()
    df = compute_metrics(args.simulator_summary, args.model_label, args.k_params)
    if args.output:
        df.to_csv(args.output, index=False)
    else:
        print(df)


if __name__ == "__main__":
    main()
