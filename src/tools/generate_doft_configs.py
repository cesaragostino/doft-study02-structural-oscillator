"""Generate DOFT simulator config JSONs from Study 01 CSV outputs.

The script reads the fingerprint and cluster CSVs that live under one of the
`results/study_01_motherfreq_thermalshift/results_*` folders and emits the
three JSON artefacts required by the DOFT cluster simulator:

* `material_config_<name>.json`
* `ground_truth_targets_<name>.json`
* `loss_weights_default_<name>.json`

Usage example:

    python scripts/tools/generate_doft_configs.py \
        --results-root results/study_01_motherfreq_thermalshift/results_w400_p7919 \
        --tag fp_kappa_w400_p7919 \
        --materials MgB2 Sn \
        --output-dir configs/study01 \
        --q-strategy-single gating \
        --eta-config results/study_01_motherfreq_thermalshift/doft_config.json

The script intentionally keeps the amount of heuristics low.  It extracts every
row for a given material/sub-network from `results_*_full_factorized.csv` and
uses the entry with the largest ``d`` (thermal → gap is usually 3) whose
exponent columns are not null as the fingerprint exponents.

`q_exp` is read directly from the same CSV when a rational lock exists.  For
``sub_network == "single"`` families missing a per-material `q`, you can choose
between two behaviours via ``--q-strategy-single``:

* `gating` (default): store `null` in `ground_truth_targets` and set `w_q=0`
  for that subnet in `loss_weights`.
* `proxy`: use the group-average `q_avg` from the bootstrap file
  (`fingerprint_*_bootstrap_CIs.csv`).

Residual targets fall back to the mean per `(category, sub_network)` in
`fingerprint_*_log_residual.csv`.  Contrast targets are taken from
`results_cluster_*` rows whose `sub_network` contains "vs".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PRIME_ORDER = [2, 3, 5, 7]
DEFAULT_CONSTRAINTS = {
    "ratios_bounds": [-0.6, 0.6],
    "deltas_bounds": [-0.8, 0.2],
    "f0_bounds": [15.0, 30.0],
}
DEFAULT_WEIGHTS = {
    "w_e": 1.0,
    "w_q": 0.5,
    "w_r": 0.25,
    "w_c": 0.3,
    "w_anchor": 0.05,
    "lambda_reg": 5e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DOFT JSON configs from Study 01 CSV sources")
    parser.add_argument("--results-root", type=Path, required=True, help="Path to results_{w}_{p} directory")
    parser.add_argument(
        "--tag",
        required=True,
        help="Common tag used in CSV file names, e.g. fp_kappa_w400_p7919",
    )
    parser.add_argument(
        "--materials",
        nargs="*",
        help="Materials to generate.  When omitted, all materials present in the dataset are processed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where JSONs will be written")
    parser.add_argument(
        "--q-strategy-single",
        choices=["gating", "proxy"],
        default="gating",
        help="How to handle q_exp for single-family materials when per-material q is missing",
    )
    parser.add_argument(
        "--eta-config",
        type=Path,
        default=Path("results/study_01_motherfreq_thermalshift/doft_config.json"),
        help="Path to doft_config.json to read the global eta value",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="Optional JSON string overriding the default constraints object",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional JSON string overriding the default loss weights",
    )
    return parser.parse_args()


def load_csv(root: Path, subdir: str, filename: str) -> pd.DataFrame:
    path = root / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def group_key(row: pd.Series) -> str:
    category = str(row.get("category", "")).strip()
    subnet = str(row.get("sub_network", "")).strip()
    return f"{category}_{subnet}" if category and subnet else ""


def pick_fingerprint_row(df: pd.DataFrame, material: str, subnet: str) -> Optional[pd.Series]:
    subset = df[(df["name"] == material) & (df["sub_network"] == subnet)]
    subset = subset.dropna(subset=["exp_a_2", "exp_b_3", "exp_c_5", "exp_d_7"], how="all")
    if subset.empty:
        return None
    subset = subset.sort_values(by=["d"], ascending=False)
    return subset.iloc[0]


def collect_materials(df: pd.DataFrame, requested: Optional[Sequence[str]]) -> List[str]:
    names = sorted(set(df["name"]))
    if not requested:
        return names
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"Materials not found in dataset: {missing}")
    return list(requested)


def build_ground_truth_entry(
    row: pd.Series,
    residual_lookup: Dict[Tuple[str, str], float],
    q_value: Optional[float],
) -> Dict[str, object]:
    exponents = [float(row.get(col, 0.0) or 0.0) for col in ("exp_a_2", "exp_b_3", "exp_c_5", "exp_d_7")]
    int_exponents = [int(round(value)) for value in exponents]
    category = str(row.get("category", ""))
    subnet = str(row.get("sub_network", ""))
    residual_value = row.get("log_residual_eta")
    if residual_value is not None and not pd.isna(residual_value):
        residual = float(residual_value)
    else:
        residual = residual_lookup.get((category, subnet), 0.0)
    return {
        "e_exp": exponents,
        "q_exp": q_value,
        "residual_exp": residual,
        "input_exponents": int_exponents,
    }


def split_contrast_label(label: str) -> Optional[Tuple[str, str]]:
    """Split strings like 'sigma-vs-pi' or 'sigma_vs_pi'."""
    parts = re.split(r"(?i)[_-]vs[_-]", label, maxsplit=1)
    if len(parts) == 2:
        left, right = parts
        return left.strip(), right.strip()
    return None


def main() -> None:
    args = parse_args()
    fingerprint_dir = args.results_root / "fingerprint"
    cluster_dir = args.results_root / "cluster"

    full_df = load_csv(fingerprint_dir, "", f"results_{args.tag}_full_factorized.csv")
    bootstrap_df = load_csv(fingerprint_dir, "", f"fingerprint_{args.tag}_bootstrap_CIs.csv")
    residual_df = load_csv(fingerprint_dir, "", f"fingerprint_{args.tag}_log_residual.csv")
    cluster_df = load_csv(cluster_dir, "", f"results_cluster_{args.tag.replace('fp_', '')}.csv")

    residual_lookup = {
        (str(row.category), str(row.sub_network)): float(row.mean)
        for row in residual_df.itertuples()
    }

    bootstrap_q = {}
    for row in bootstrap_df.itertuples():
        metric = str(row.metric)
        if metric != "q_avg":
            continue
        bootstrap_q[(str(row.group))] = float(row.mean)

    materials = collect_materials(full_df, args.materials)
    eta_value = DEFAULT_WEIGHTS["lambda_reg"]
    if args.eta_config and Path(args.eta_config).exists():
        eta_data = json.loads(Path(args.eta_config).read_text())
        eta_value = float(eta_data.get("CALIBRATED_ETA", eta_value))

    constraints = json.loads(args.constraints) if args.constraints else DEFAULT_CONSTRAINTS
    weight_template = json.loads(args.weights) if args.weights else DEFAULT_WEIGHTS

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows = cluster_df[cluster_df["sub_network"].str.contains("vs", case=False, na=False)]

    for material in materials:
        material_rows = full_df[full_df["name"] == material]
        subnets = sorted({str(s) for s in material_rows["sub_network"] if "vs" not in str(s)})
        anchors = {}
        ground_truth = {}
        q_overrides: Dict[str, float] = {}

        for subnet in subnets:
            row = pick_fingerprint_row(full_df, material, subnet)
            if row is None:
                continue
            key = f"{material}_{subnet}"
            q_value = None
            rational_rows = material_rows[
                (material_rows["sub_network"] == subnet)
                & material_rows["lock_family"].str.contains("rational", case=False, na=False)
                & material_rows["q"].notna()
            ]
            if not rational_rows.empty:
                q_value = float(rational_rows.iloc[0]["q"])

            is_single_family = subnet == "single"
            if q_value is None and is_single_family and args.q_strategy_single == "proxy":
                proxy = bootstrap_q.get(group_key(row))
                if proxy is not None:
                    q_value = proxy

            if q_value is None and (not is_single_family or args.q_strategy_single == "gating"):
                q_overrides[key] = 0.0

            ground_truth[key] = build_ground_truth_entry(row, residual_lookup, q_value)
            x_value = float(row.get("X", 0.0) or 0.0)
            anchors[subnet] = {"f0": x_value, "X": x_value}

        # contrasts
        material_contrasts: List[Dict[str, object]] = []
        mat_cluster = cluster_rows[cluster_rows["name"] == material]
        for row in mat_cluster.itertuples():
            label = str(row.sub_network)
            split = split_contrast_label(label)
            if not split:
                print(f"[WARN] Could not parse contrast label '{label}' for {material}; skipping")
                continue
            A, B = split
            material_contrasts.append(
                {
                    "type": label,
                    "A": A,
                    "B": B,
                    "C_AB_exp": float(row.R_obs),
                }
            )

        material_config = {
            "material": material,
            "subnetworks": subnets,
            "anchors": anchors,
            "primes": {str(p): {"layer": 1} for p in PRIME_ORDER},
            "constraints": constraints,
            "freeze_primes": [],
            "layers": {subnet: 1 for subnet in subnets},
            "eta": eta_value,
        }
        if material_contrasts:
            material_config["contrasts"] = material_contrasts

        wt = dict(weight_template)
        overrides = {}
        if q_overrides:
            overrides["q"] = q_overrides
        if overrides:
            wt["overrides"] = overrides

        output_base = args.output_dir / f"{material.lower()}"
        output_base.parent.mkdir(parents=True, exist_ok=True)

        (args.output_dir / f"material_config_{material}.json").write_text(json.dumps(material_config, indent=2))
        (args.output_dir / f"ground_truth_targets_{material}.json").write_text(json.dumps(ground_truth, indent=2))
        (args.output_dir / f"loss_weights_default_{material}.json").write_text(json.dumps(wt, indent=2))
        print(f"Generated configs for {material} -> {args.output_dir}")


if __name__ == "__main__":
    main()
