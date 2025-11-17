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


def build_sim_digest(runs_dir: Path, digest_dir: Path) -> None:
    import json
    import pandas as pd

    rows = []
    for manifest_path in runs_dir.rglob("manifest.json"):
        with manifest_path.open() as fh:
            data = json.load(fh)
        rows.append(
            {
                "material": data.get("material"),
                "seed": data.get("seed"),
                "max_evals": data.get("max_evals"),
                "total_loss": data.get("total_loss"),
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


def inject_xi_into_configs(configs_dir: Path, xi_map: Dict[str, float]) -> None:
    """Add xi value and xi_sign mapping to each material_config_*.json."""
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

    for cfg_path in configs_dir.glob("material_config_*.json"):
        mat = cfg_path.stem.replace("material_config_", "")
        xi_value = xi_map.get(mat)
        if xi_value is None:
            continue
        try:
            data = json.loads(cfg_path.read_text())
        except Exception:
            continue
        subnets = data.get("subnets", [])
        if not subnets:
            continue
        xi_sign = derive_signs([str(s) for s in subnets])
        data["xi"] = float(xi_value)
        data["xi_sign"] = xi_sign
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
        ]
        if args.fit_noise_by_category:
            noise_cmd.append("--fit-by-category")
        if materials:
            noise_cmd += ["--materials", *materials]
        run_cmd(noise_cmd, cwd=Path("."))
        if noise_json.exists():
            import json as _json

            xi_map = _json.loads(noise_json.read_text())

    # Inject xi/xip signs into material configs
    if xi_map:
        inject_xi_into_configs(configs_dir, xi_map)

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

    # 3) Structural noise computation
    if not args.skip_noise:
        noise_csv = noise_dir / "structural_noise_summary.csv"
        noise_json = noise_dir / "structural_noise_values.json"
        noise_cmd = [
            "python3",
            "src/compute_structural_noise.py",
            "--materials-csv",
            str(args.materials_csv),
            "--output-csv",
            str(noise_csv),
            "--output-json",
            str(noise_json),
        ]
        if args.fit_noise_by_category:
            noise_cmd.append("--fit-by-category")
        if materials:
            noise_cmd += ["--materials", *materials]
        run_cmd(noise_cmd, cwd=Path("."))

    # 4) Build digests for configs, simulator, and structural noise
    build_config_digest(configs_dir, digest_dir, args.materials_csv)
    if not args.skip_simulator:
        build_sim_digest(runs_dir, digest_dir)
    if not args.skip_noise:
        copy_noise_digest(noise_csv, digest_dir)

    print(f"Pipeline completed. Outputs under: {output_root}")


if __name__ == "__main__":
    main()
