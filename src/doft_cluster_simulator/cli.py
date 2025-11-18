"""Command line interface for the DOFT cluster simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .data import MaterialConfig, ParameterBounds, TargetDataset, load_loss_weights
from .engine import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DOFT cluster simulator")
    parser.add_argument("--config", type=Path, required=True, help="Ruta al archivo material_config.json")
    parser.add_argument("--targets", type=Path, required=True, help="Ruta al archivo de targets ground-truth")
    parser.add_argument("--weights", type=Path, help="Archivo JSON opcional con pesos de pérdida")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Directorio de salida")
    parser.add_argument("--max-evals", type=int, default=500, help="Evaluaciones máximas por subred")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base de RNG")
    parser.add_argument("--seed-sweep", type=int, default=1, help="Cantidad de semillas a evaluar")
    parser.add_argument(
        "--bounds",
        nargs="*",
        help="Overrides de límites (ej: ratios=-0.2,0.2 deltas=-0.3,0.3 f0=18,22)",
    )
    parser.add_argument("--huber-delta", type=float, default=None, help="Delta para la pérdida Huber (opcional)")
    parser.add_argument("--anchor-weight", type=float, help="Override rápido de w_anchor")
    return parser


def run_from_args(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = MaterialConfig.from_file(args.config)
    dataset = TargetDataset.from_file(args.targets)
    weights = load_loss_weights(args.weights)
    if args.anchor_weight is not None:
        weights.w_anchor = float(args.anchor_weight)

    bounds_override = _parse_bounds_override(args.bounds)
    bounds = ParameterBounds.from_cli(bounds_override)
    seed_sweep = max(1, args.seed_sweep)
    sweep_results = []

    for idx in range(seed_sweep):
        seed_value = args.seed + idx
        engine = SimulationEngine(
            config=config,
            dataset=dataset,
            weights=weights,
            max_evals=args.max_evals,
            seed=seed_value,
            bounds=bounds,
            huber_delta=args.huber_delta,
        )
        bundle = engine.run()
        run_outdir = args.outdir if seed_sweep == 1 else args.outdir / f"seed_{seed_value}"
        extras = {
            "huber_delta": args.huber_delta,
            "bounds": bounds_override,
            "seed_sweep": seed_sweep,
        }
        bundle.write(run_outdir, args.config, args.targets, args.max_evals, seed_value, extras=extras)
        sweep_results.append({"seed": seed_value, "outdir": str(run_outdir), "total_loss": bundle.total_loss})

    if seed_sweep > 1:
        args.outdir.mkdir(parents=True, exist_ok=True)
        best = min(sweep_results, key=lambda item: item["total_loss"])
        summary_path = args.outdir / "sweep_manifest.json"
        summary_path.write_text(json.dumps({"runs": sweep_results, "best_seed": best["seed"], "best_outdir": best["outdir"]}, indent=2))


def main() -> None:
    run_from_args()


def _parse_bounds_override(entries: Optional[list[str]]) -> Optional[dict[str, tuple[float, float]]]:
    if not entries:
        return None
    data: dict[str, tuple[float, float]] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        key, raw_values = entry.split("=", 1)
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if len(values) != 2:
            continue
        try:
            pair = (float(values[0]), float(values[1]))
        except ValueError:
            continue
        key_name = key.strip().lower()
        if key_name in {"ratios", "ratio", "ratios_bounds"}:
            data["ratios_bounds"] = pair
        elif key_name in {"deltas", "delta", "deltas_bounds"}:
            data["deltas_bounds"] = pair
        elif key_name in {"f0", "f0_bounds"}:
            data["f0_bounds"] = pair
        elif key_name in {"delta_t", "delta_t_bounds"}:
            data["delta_T_bounds"] = pair
        elif key_name in {"delta_space", "delta_space_bounds", "deltaspace"}:
            data["delta_space_bounds"] = pair
    return data or None


if __name__ == "__main__":
    main()
