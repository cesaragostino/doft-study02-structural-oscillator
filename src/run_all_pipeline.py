"""Run the full DOFT pipeline (configs → simulator → structural noise) in one go.

Example:

python3 src/run_all_pipeline.py \
  --results-root data/raw/fingerprint-run2-v6/results_w800_p7919 \
  --tag fp_kappa_w800_p7919 \
  --materials Al Hf Mo Nb NbN Re Ta Ti V Zr \
  --output-root outputs/run_w800_p7919 \
  --bounds ratios=-0.25,0.25 deltas=-0.35,0.35 f0=12,500 \
  --huber-delta 0.02 \
  --max-evals 1200 \
  --seed 123 \
  --seed-sweep 5 \
  --fit-noise-by-category
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

from doft_cluster_simulator.data import PRIMES


def run_cmd(cmd: List[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def parse_bounds(bounds: Optional[List[str]]) -> List[str]:
    return bounds or []


def detect_materials_from_configs(config_dir: Path) -> List[str]:
    mats = []
    for path in config_dir.glob("material_config_*.json"):
        name = path.stem.replace("material_config_", "")
        if name:
            mats.append(name)
    return sorted(mats)


def default_eta_path(results_root: Path, tag: str) -> Path:
    """Infer eta path from results_root and tag (strip fp_ and optional kappa_)."""
    calib_dir = results_root / "calib"
    core = tag
    if core.startswith("fp_"):
        core = core[len("fp_") :]
    if core.startswith("kappa_"):
        core = core[len("kappa_") :]
    filename = f"calibration_metadata_calib_{core}.json"
    return calib_dir / filename


def load_all_materials(materials_csv: Path) -> List[str]:
    import pandas as pd

    if not materials_csv.exists():
        return []
    df = pd.read_csv(materials_csv)
    if "name" not in df.columns:
        return []
    return sorted(df["name"].dropna().astype(str).unique().tolist())


def build_sim_digest(runs_dir: Path, digest_dir: Path, pressure_lookup: Optional[Dict[str, float]] = None) -> None:
    import json
    import pandas as pd

    rows = []
    for manifest_path in runs_dir.rglob("manifest.json"):
        with manifest_path.open() as fh:
            data = json.load(fh)
        delta_T_values = {str(p): [] for p in PRIMES}
        delta_space_values = {str(p): [] for p in PRIMES}
        delta_P_values = {str(p): [] for p in PRIMES}
        material_name = data.get("material")
        best_params_path = manifest_path.parent / "best_params.json"
        if best_params_path.exists():
            try:
                best = json.loads(best_params_path.read_text())
                params = best.get("params", {})
                for val in params.values():
                    dt = val.get("delta_T")
                    ds = val.get("delta_space")
                    if isinstance(dt, (int, float)):
                        for prime in PRIMES:
                            delta_T_values[str(prime)].append(float(dt))
                    elif isinstance(dt, dict):
                        for k, v in dt.items():
                            key = str(k)
                            if key in delta_T_values and isinstance(v, (int, float)):
                                delta_T_values[key].append(float(v))
                    if isinstance(ds, (int, float)):
                        for prime in PRIMES:
                            delta_space_values[str(prime)].append(float(ds))
                    elif isinstance(ds, dict):
                        for k, v in ds.items():
                            key = str(k)
                            if key in delta_space_values and isinstance(v, (int, float)):
                                delta_space_values[key].append(float(v))
                    dp = val.get("delta_P")
                    if isinstance(dp, (int, float)):
                        for prime in PRIMES:
                            delta_P_values[str(prime)].append(float(dp))
                    elif isinstance(dp, dict):
                        for k, v in dp.items():
                            key = str(k)
                            if key in delta_P_values and isinstance(v, (int, float)):
                                delta_P_values[key].append(float(v))
            except Exception:
                pass
        delta_T_mean = {f"delta_T_mean_{p}": (sum(vals) / len(vals)) if vals else None for p, vals in delta_T_values.items()}
        delta_T_min = {f"delta_T_min_{p}": (min(vals) if vals else None) for p, vals in delta_T_values.items()}
        delta_T_max = {f"delta_T_max_{p}": (max(vals) if vals else None) for p, vals in delta_T_values.items()}
        delta_space_mean = {f"delta_space_mean_{p}": (sum(vals) / len(vals)) if vals else None for p, vals in delta_space_values.items()}
        delta_space_min = {f"delta_space_min_{p}": (min(vals) if vals else None) for p, vals in delta_space_values.items()}
        delta_space_max = {f"delta_space_max_{p}": (max(vals) if vals else None) for p, vals in delta_space_values.items()}
        delta_P_mean = {f"delta_P_mean_{p}": (sum(vals) / len(vals)) if vals else None for p, vals in delta_P_values.items()}
        delta_P_min = {f"delta_P_min_{p}": (min(vals) if vals else None) for p, vals in delta_P_values.items()}
        delta_P_max = {f"delta_P_max_{p}": (max(vals) if vals else None) for p, vals in delta_P_values.items()}
        pressure_val = None
        if pressure_lookup and material_name in pressure_lookup:
            pressure_val = pressure_lookup[material_name]
        rows.append(
            {
                "material": material_name,
                "seed": data.get("seed"),
                "max_evals": data.get("max_evals"),
                "total_loss": data.get("total_loss"),
                **delta_T_mean,
                **delta_T_min,
                **delta_T_max,
                **delta_space_mean,
                **delta_space_min,
                **delta_space_max,
                **delta_P_mean,
                **delta_P_min,
                **delta_P_max,
                "pressure_GPa": pressure_val,
                "weights_w_e": data.get("weights", {}).get("w_e"),
                "weights_w_q": data.get("weights", {}).get("w_q"),
                "weights_w_r": data.get("weights", {}).get("w_r"),
                "weights_w_c": data.get("weights", {}).get("w_c"),
                "weights_w_anchor": data.get("weights", {}).get("w_anchor"),
                "huber_delta": data.get("extras", {}).get("huber_delta"),
                "bounds": data.get("extras", {}).get("bounds"),
                "seed_sweep": data.get("extras", {}).get("seed_sweep"),
                "path": str(manifest_path.parent),
            }
        )
    if not rows:
        return
    digest_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(digest_dir / "simulator_summary.csv", index=False)
    try:
        df.to_excel(digest_dir / "simulator_summary.xlsx", index=False)
    except Exception:
        pass
    value_cols = [col for col in df.columns if col.startswith("delta_T_") or col.startswith("delta_space_") or col.startswith("delta_P_")]
    if value_cols:
        material_summary = df.groupby("material")[value_cols].agg(["mean", "min", "max"])
        material_summary.columns = [f"{col}_{stat}" for col, stat in material_summary.columns]
        material_summary.reset_index(inplace=True)
        material_summary.to_csv(digest_dir / "simulator_delta_by_material.csv", index=False)
        config_meta_path = digest_dir / "config_fingerprint_summary.csv"
        if config_meta_path.exists():
            try:
                meta_df = pd.read_csv(config_meta_path)[["material", "category"]].drop_duplicates()
                df_with_cat = df.merge(meta_df, on="material", how="left")
                category_summary = df_with_cat.groupby("category")[value_cols].agg(["mean", "min", "max"])
                category_summary.columns = [f"{col}_{stat}" for col, stat in category_summary.columns]
                category_summary.reset_index(inplace=True)
                category_summary.to_csv(digest_dir / "simulator_delta_by_category.csv", index=False)
            except Exception:
                pass


def build_pressure_digest(noise_json: Path, sim_summary: Path, digest_dir: Path) -> None:
    """Summarise pressure inputs and resulting delta_P values."""
    import json
    import pandas as pd
    import numpy as np

    if not noise_json.exists() or not sim_summary.exists():
        return
    data = json.loads(noise_json.read_text())
    pressure_map = {}
    for name, entry in data.items():
        pressure_map[name] = float(entry.get("pressure_GPa", 0.0) or 0.0)
    sim_df = pd.read_csv(sim_summary)
    delta_cols = [col for col in sim_df.columns if col.startswith("delta_P_mean_")]
    if not delta_cols:
        return
    agg = sim_df.groupby("material")[["total_loss"] + delta_cols].mean().reset_index()
    agg["pressure_GPa"] = agg["material"].map(pressure_map).fillna(0.0)
    # Correlation of pressure vs loss and deltas across materials
    if len(agg) >= 2:
        agg["corr_pressure_loss"] = np.corrcoef(agg["pressure_GPa"], agg["total_loss"])[0, 1] if agg["pressure_GPa"].std() > 0 else np.nan
    else:
        agg["corr_pressure_loss"] = np.nan
    digest_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(digest_dir / "simulator_pressure_by_material.csv", index=False)


def build_model_selection_metrics(sim_summary: Path, digest_dir: Path, model_label: str = "vector_pressure", k_params: int = 21) -> None:
    import math
    import pandas as pd

    if not sim_summary.exists():
        return
    df = pd.read_csv(sim_summary)
    if df.empty or "total_loss" not in df.columns:
        return
    n = len(df)
    loss_sum = df["total_loss"].sum()
    aic = 2 * k_params + loss_sum
    bic = k_params * math.log(max(n, 1)) + loss_sum
    out_df = pd.DataFrame(
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
    digest_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(digest_dir / "model_selection_summary.csv", index=False)


def copy_noise_digest(noise_csv: Path, digest_dir: Path) -> None:
    import pandas as pd

    if not noise_csv.exists():
        return
    digest_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(noise_csv)
    df.to_csv(digest_dir / "structural_noise_summary.csv", index=False)
    try:
        df.to_excel(digest_dir / "structural_noise_summary.xlsx", index=False)
    except Exception:
        pass


def build_family_correlation(noise_csv: Path, sim_summary: Path, digest_dir: Path) -> None:
    """Aggregate δT/δspace behaviour by family/category and correlate vs predicted_noise."""
    import pandas as pd
    import numpy as np

    if not noise_csv.exists() or not sim_summary.exists():
        return
    noise_df = pd.read_csv(noise_csv)
    noise_df["material"] = noise_df["name"].astype(str)
    noise_df["family"] = noise_df.get("category", "Unknown").fillna("Unknown").astype(str)
    sim_df = pd.read_csv(sim_summary)
    delta_mean_cols = [
        col for col in sim_df.columns if col.startswith("delta_T_mean_") or col.startswith("delta_space_mean_") or col.startswith("delta_P_mean_")
    ]
    if not delta_mean_cols:
        return
    sim_mat = sim_df.groupby("material")[delta_mean_cols].mean().reset_index()
    merged = sim_mat.merge(noise_df[["material", "predicted_noise", "mismatch_mean", "M_struct_mean", "family"]], on="material", how="left")
    merged["family"] = merged["family"].fillna("Unknown")
    rows = []

    def _corr(a: pd.Series, b: pd.Series) -> float | None:
        paired = pd.concat([a, b], axis=1).dropna()
        if len(paired) < 2:
            return None
        cval = paired.iloc[:, 0].corr(paired.iloc[:, 1])
        return float(cval) if pd.notna(cval) else None

    for family, g in merged.groupby("family"):
        if len(g) == 0:
            continue
        entry = {
            "family": family,
            "material_count": len(g),
            "predicted_noise_mean": g["predicted_noise"].mean(),
            "mismatch_mean": g["mismatch_mean"].mean(),
            "M_struct_mean": g["M_struct_mean"].mean(),
        }
        for prime in PRIMES:
            dt_col = f"delta_T_mean_{prime}"
            ds_col = f"delta_space_mean_{prime}"
            if dt_col in g.columns:
                entry[f"{dt_col}_mean"] = g[dt_col].mean()
                entry[f"corr_noise_{dt_col}"] = _corr(g["predicted_noise"], g[dt_col])
            if ds_col in g.columns:
                entry[f"{ds_col}_mean"] = g[ds_col].mean()
                entry[f"corr_noise_{ds_col}"] = _corr(g["predicted_noise"], g[ds_col])
        rows.append(entry)
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df.sort_values(by="family", inplace=True)
    digest_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(digest_dir / "family_correlation_summary.csv", index=False)


def build_config_digest(configs_dir: Path, digest_dir: Path, materials_csv: Path) -> None:
    """Aggregate per-material fingerprint targets/anchors into a single CSV/XLSX."""
    import json
    import pandas as pd

    meta_by_key = {}
    if materials_csv.exists():
        df_meta = pd.read_csv(materials_csv)
        for row in df_meta.itertuples():
            name = str(getattr(row, "name"))
            subnet = str(getattr(row, "sub_network")) if hasattr(row, "sub_network") else None
            key = (name, subnet) if subnet is not None else (name, "")
            meta_by_key[key] = {
                "category": getattr(row, "category", None),
                "Tc_K": getattr(row, "Tc_K", None),
                "Gap_meV": getattr(row, "Gap_meV", None),
                "ThetaD_K": getattr(row, "ThetaD_K", None),
                "EF_eV": getattr(row, "EF_eV", None),
            }

    rows = []
    for gt_path in configs_dir.glob("ground_truth_targets_*.json"):
        mat = gt_path.stem.replace("ground_truth_targets_", "")
        cfg_path = configs_dir / f"material_config_{mat}.json"
        anchors = {}
        if cfg_path.exists():
            cfg_data = json.loads(cfg_path.read_text())
            anchors = cfg_data.get("anchors", {})

        data = json.loads(gt_path.read_text())
        for key, val in data.items():
            if "_vs_" in key:
                continue
            subnet = key
            prefix = f"{mat}_"
            if subnet.startswith(prefix):
                subnet = subnet[len(prefix) :]
            e_list = val.get("e_exp") or [None, None, None, None]

            meta = meta_by_key.get((mat, subnet)) or meta_by_key.get((mat, "")) or {}
            rows.append(
                {
                    "material": mat,
                    "subnet": subnet,
                    "category": meta.get("category"),
                    "Tc_K": meta.get("Tc_K"),
                    "Gap_meV": meta.get("Gap_meV"),
                    "ThetaD_K": meta.get("ThetaD_K"),
                    "EF_eV": meta.get("EF_eV"),
                    "e2": e_list[0],
                    "e3": e_list[1],
                    "e5": e_list[2],
                    "e7": e_list[3],
                    "q_exp": val.get("q_exp"),
                    "residual_exp": val.get("residual_exp"),
                    "anchor_f0": anchors.get(subnet, {}).get("f0"),
                    "anchor_X": anchors.get(subnet, {}).get("X"),
                }
            )

    if not rows:
        return
    digest_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(digest_dir / "config_fingerprint_summary.csv", index=False)
    try:
        df.to_excel(digest_dir / "config_fingerprint_summary.xlsx", index=False)
    except Exception:
        pass


def inject_xi_into_configs(
    configs_dir: Path, xi_map: Dict[str, float], default_delta_T: float, default_delta_space: float
) -> None:
    """Add xi value/xi_exp/k_skin and xi_sign mapping to each material_config_*.json."""
    import json

    def derive_signs(subnets: List[str]) -> Dict[str, int]:
        signs: Dict[str, int] = {}
        # Prefer semantic mapping when sigma/pi present
        for subnet in subnets:
            lower = subnet.lower()
            if "sigma" in lower:
                signs[subnet] = 1
            elif "pi" in lower:
                signs[subnet] = -1
        # If none set or incomplete, assign alternating signs
        if not signs:
            alt = [1, -1]
            for idx, subnet in enumerate(subnets):
                signs[subnet] = alt[idx % 2]
        else:
            # Fill any missing with alternating pattern continuing current size
            idx = 0
            for subnet in subnets:
                if subnet in signs:
                    continue
                signs[subnet] = 1 if idx % 2 == 0 else -1
                idx += 1
        return signs

    def build_prime_vector(raw: object, fallback_scalar: float) -> dict:
        base = {str(p): float(fallback_scalar) for p in PRIMES}
        if isinstance(raw, (int, float)):
            return {str(p): float(raw) for p in PRIMES}
        if isinstance(raw, dict):
            for k, v in raw.items():
                key = str(k)
                if key in base and isinstance(v, (int, float)):
                    base[key] = float(v)
        return base

    def parse_delta_block(raw: object, subnets: List[str], fallback_scalar: float) -> dict:
        shared_default = build_prime_vector(raw if isinstance(raw, (int, float)) else fallback_scalar, fallback_scalar)
        result = {subnet: dict(shared_default) for subnet in subnets}
        if isinstance(raw, dict):
            has_subnet_keys = any(str(k) in raw for k in subnets)
            if has_subnet_keys:
                for subnet in subnets:
                    if subnet in raw:
                        result[subnet] = build_prime_vector(raw[subnet], fallback_scalar)
                return result
            shared_default = build_prime_vector(raw, fallback_scalar)
            return {subnet: dict(shared_default) for subnet in subnets}
        return result

    def infer_lambda_band(subnet: str) -> float:
        lower = subnet.lower()
        if "pi" in lower:
            return 0.8
        if "sigma" in lower:
            return 1.0
        return 1.0

    def parse_lambda_pressure_band(raw: object, subnets: List[str], base_map: Dict[str, float]) -> Dict[str, float]:
        result = dict(base_map)
        if isinstance(raw, (int, float)):
            return {subnet: float(raw) for subnet in subnets}
        if isinstance(raw, dict):
            for subnet in subnets:
                val = raw.get(subnet)
                if isinstance(val, (int, float)):
                    result[subnet] = float(val)
        return result

    for cfg_path in configs_dir.glob("material_config_*.json"):
        mat = cfg_path.stem.replace("material_config_", "")
        xi_entry = xi_map.get(mat)
        if xi_entry is None:
            continue
        delta_T_entry = None
        delta_space_entry = None
        delta_P_entry = None
        lambda_band_entry: dict = {}
        lambda_geo_entry: dict = {}
        lambda_pressure_band_entry: dict = {}
        lambda_pressure_geo_entry: dict = {}
        # Backward compatibility: allow xi_entry to be a scalar
        if isinstance(xi_entry, (int, float)):
            xi_value = float(xi_entry)
            xi_exp = {}
            k_skin = 0.0
        elif isinstance(xi_entry, dict):
            xi_value = xi_entry.get("xi")
            xi_exp = xi_entry.get("xi_exp", {})
            k_skin = float(xi_entry.get("k_skin", 0.0))
            delta_T_entry = xi_entry.get("delta_T", None)
            delta_space_entry = xi_entry.get("delta_space", None)
            delta_P_entry = xi_entry.get("delta_P", None)
            lambda_band_entry = xi_entry.get("lambda_band", {})
            lambda_geo_entry = xi_entry.get("lambda_geo", {})
            lambda_pressure_band_entry = xi_entry.get("lambda_pressure_band", {})
            lambda_pressure_geo_entry = xi_entry.get("lambda_pressure_geo", {})
        else:
            continue
        if xi_value is None:
            continue
        try:
            data = json.loads(cfg_path.read_text())
        except Exception:
            continue
        subnets = data.get("subnets", [])
        if not subnets:
            continue
        if "category" not in data and isinstance(xi_entry, dict):
            if "category" in xi_entry:
                data["category"] = xi_entry["category"]
        xi_sign = derive_signs([str(s) for s in subnets])
        config_delta_T = parse_delta_block(data.get("delta_T"), subnets, default_delta_T)
        config_delta_space = parse_delta_block(data.get("delta_space"), subnets, default_delta_space)
        config_delta_P = parse_delta_block(data.get("delta_P"), subnets, 0.0)
        delta_T_map = parse_delta_block(delta_T_entry, subnets, default_delta_T) if delta_T_entry is not None else config_delta_T
        delta_space_map = parse_delta_block(delta_space_entry, subnets, default_delta_space) if delta_space_entry is not None else config_delta_space
        delta_P_map = parse_delta_block(delta_P_entry, subnets, 0.0) if delta_P_entry is not None else config_delta_P
        lambda_band_map = {subnet: infer_lambda_band(subnet) for subnet in subnets}
        if isinstance(data.get("lambda_band"), dict):
            for subnet, value in data["lambda_band"].items():
                if subnet in lambda_band_map and isinstance(value, (int, float)):
                    lambda_band_map[subnet] = float(value)
        if isinstance(lambda_band_entry, dict):
            for subnet, value in lambda_band_entry.items():
                if subnet in lambda_band_map and isinstance(value, (int, float)):
                    lambda_band_map[subnet] = float(value)
        lambda_geo_map = build_prime_vector(lambda_geo_entry, 1.0)
        if isinstance(data.get("lambda_geo"), dict):
            for k, v in data["lambda_geo"].items():
                if str(k) in lambda_geo_map and isinstance(v, (int, float)):
                    lambda_geo_map[str(k)] = float(v)
        lambda_pressure_band_map = parse_lambda_pressure_band(lambda_pressure_band_entry, subnets, lambda_band_map)
        lambda_pressure_geo_map = build_prime_vector(lambda_pressure_geo_entry, 1.0)
        if isinstance(data.get("lambda_pressure_geo"), dict):
            for k, v in data["lambda_pressure_geo"].items():
                if str(k) in lambda_pressure_geo_map and isinstance(v, (int, float)):
                    lambda_pressure_geo_map[str(k)] = float(v)
        data["xi"] = float(xi_value)
        data["xi_sign"] = xi_sign
        if xi_exp:
            data["xi_exp"] = xi_exp
        if "k_skin" not in data:
            data["k_skin"] = k_skin
        else:
            data["k_skin"] = data.get("k_skin", k_skin)
        data["delta_T"] = delta_T_map
        data["delta_space"] = delta_space_map
        data["delta_P"] = delta_P_map
        data["lambda_band"] = lambda_band_map
        data["lambda_geo"] = lambda_geo_map
        data["lambda_pressure_band"] = lambda_pressure_band_map
        data["lambda_pressure_geo"] = lambda_pressure_geo_map
        cfg_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full DOFT pipeline end-to-end.")
    parser.add_argument("--results-root", type=Path, required=True, help="Fingerprint results root (results_w*_p*)")
    parser.add_argument("--tag", required=True, help="Tag used in CSV filenames, e.g. fp_kappa_w800_p7919")
    parser.add_argument(
        "--materials",
        nargs="*",
        help="Materials to process; use 'all' to load every name from materials-csv; omit to auto-detect from generated configs",
    )
    parser.add_argument("--eta", type=Path, help="Calibration metadata JSON with eta (auto-resolved if omitted)")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/run_all"), help="Root folder for all artifacts")
    parser.add_argument("--q-strategy-single", choices=["gating", "proxy"], default="gating")
    parser.add_argument("--bounds", nargs="*", help="Bounds override passed to simulator (ratios=.. deltas=.. f0=..)")
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--max-evals", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-sweep", type=int, default=1)
    parser.add_argument("--materials-csv", type=Path, default=Path("data/raw/materials_clusters_real_v6.csv"))
    parser.add_argument("--fit-noise-by-category", action="store_true")
    parser.add_argument("--skip-simulator", action="store_true", help="Only generate configs and structural noise")
    parser.add_argument("--skip-noise", action="store_true", help="Skip structural noise computation")
    parser.add_argument("--k-skin", type=float, default=0.05, help="Skin coupling coefficient for structural noise term")
    parser.add_argument(
        "--default-delta-T",
        type=float,
        default=0.0,
        help="Default surface temperature gradient per material (passed to structural noise calc)",
    )
    parser.add_argument(
        "--default-delta-space",
        type=float,
        default=0.0,
        help="Default spatial expansion per material (passed to structural noise calc)",
    )
    parser.add_argument(
        "--c-pressure",
        type=float,
        default=0.01,
        help="Global factor for pressure-induced delta_P initialisation",
    )
    parser.add_argument(
        "--pressure-ref",
        type=float,
        default=1.0,
        help="Reference pressure (GPa) for delta_P normalisation",
    )
    args = parser.parse_args()

    output_root = args.output_root
    configs_dir = output_root / "configs"
    runs_dir = output_root / "runs"
    noise_dir = output_root / "structural_noise"
    digest_dir = output_root / "digest"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_noise:
        noise_dir.mkdir(parents=True, exist_ok=True)

    eta_path = args.eta if args.eta else default_eta_path(args.results_root, args.tag)

    # Resolve materials before generating configs
    materials_cli = args.materials or []
    if "all" in materials_cli:
        material_list = load_all_materials(args.materials_csv)
        if not material_list:
            material_list = None
    elif materials_cli:
        material_list = materials_cli
    else:
        material_list = None  # let generator detect

    # 1) Generate DOFT configs
    gen_cmd = [
        "python3",
        "src/tools/generate_doft_configs.py",
        "--results-root",
        str(args.results_root),
        "--tag",
        args.tag,
        "--output-dir",
        str(configs_dir),
        "--q-strategy-single",
        args.q_strategy_single,
        "--eta",
        str(eta_path),
    ]
    if material_list:
        gen_cmd += ["--materials", *material_list]
    run_cmd(gen_cmd, cwd=Path("."))

    # Discover materials based on generated configs to avoid missing targets
    available_materials = detect_materials_from_configs(configs_dir)
    if material_list is not None:
        materials = [m for m in material_list if m in available_materials]
    else:
        materials = available_materials

    # 2) Structural noise computation (needed for xi injection)
    noise_csv = noise_dir / "structural_noise_summary.csv"
    noise_json = noise_dir / "structural_noise_values.json"
    xi_map = {}
    if not args.skip_noise:
        noise_cmd = [
            "python3",
            "src/compute_structural_noise.py",
            "--materials-csv",
            str(args.materials_csv),
            "--output-csv",
            str(noise_csv),
            "--output-json",
            str(noise_json),
            "--k-skin",
            str(args.k_skin),
        ]
        if args.fit_noise_by_category:
            noise_cmd.append("--fit-by-category")
        if materials:
            noise_cmd += ["--materials", *materials]
        noise_cmd += ["--default-delta-T", str(args.default_delta_T)]
        noise_cmd += ["--default-delta-space", str(args.default_delta_space)]
        noise_cmd += ["--c-pressure", str(args.c_pressure)]
        noise_cmd += ["--pressure-ref", str(args.pressure_ref)]
        run_cmd(noise_cmd, cwd=Path("."))
        if noise_json.exists():
            import json as _json

            xi_map = _json.loads(noise_json.read_text())

    # Inject xi/xip signs into material configs
    if xi_map:
        inject_xi_into_configs(configs_dir, xi_map, args.default_delta_T, args.default_delta_space)

    # 3) Run simulator per material
    if not args.skip_simulator:
        for mat in materials:
            config_path = configs_dir / f"material_config_{mat}.json"
            targets_path = configs_dir / f"ground_truth_targets_{mat}.json"
            weights_path = configs_dir / f"loss_weights_default_{mat}.json"
            if not config_path.exists() or not targets_path.exists() or not weights_path.exists():
                print(f"[WARN] Missing files for {mat}; skipping simulator run")
                continue
            outdir = runs_dir / mat.lower()
            sim_cmd = [
                "python3",
                "-m",
                "src.doft_cluster_simulator.cli",
                "--config",
                str(config_path),
                "--targets",
                str(targets_path),
                "--weights",
                str(weights_path),
                "--outdir",
                str(outdir),
                "--max-evals",
                str(args.max_evals),
                "--seed",
                str(args.seed),
                "--seed-sweep",
                str(args.seed_sweep),
            ]
            if args.bounds:
                sim_cmd += ["--bounds", *parse_bounds(args.bounds)]
            if args.huber_delta is not None:
                sim_cmd += ["--huber-delta", str(args.huber_delta)]
            run_cmd(sim_cmd, cwd=Path("."))

    # 4) Build digests for configs, simulator, and structural noise
    build_config_digest(configs_dir, digest_dir, args.materials_csv)
    sim_summary_path = digest_dir / "simulator_summary.csv"
    if not args.skip_simulator:
        build_sim_digest(runs_dir, digest_dir, {m: (xi_map.get(m, {}).get("pressure_GPa", 0.0) if isinstance(xi_map.get(m), dict) else 0.0) for m in xi_map} if xi_map else None)
        build_model_selection_metrics(sim_summary_path, digest_dir, model_label="vector_pressure")
    if not args.skip_noise:
        copy_noise_digest(noise_csv, digest_dir)
        build_family_correlation(noise_csv, sim_summary_path, digest_dir)
        build_pressure_digest(noise_json, sim_summary_path, digest_dir)

    print(f"Pipeline completed. Outputs under: {output_root}")


if __name__ == "__main__":
    main()
