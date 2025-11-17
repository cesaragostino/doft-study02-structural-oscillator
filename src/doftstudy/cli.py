"""Command line interface for the DOFT cluster simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
<<<<<<< ours
<<<<<<< ours
from typing import Dict, List, Optional, Sequence, Tuple
=======
from typing import Optional
>>>>>>> theirs
=======
from typing import Optional
>>>>>>> theirs

from .data import MaterialConfig, TargetDataset, load_loss_weights
from .engine import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DOFT cluster simulator")
    parser.add_argument("--config", type=Path, required=True, help="Ruta al archivo material_config.json")
    parser.add_argument("--targets", type=Path, required=True, help="Ruta al archivo de targets ground-truth")
    parser.add_argument("--weights", type=Path, help="Archivo JSON opcional con pesos de pérdida")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Directorio de salida")
<<<<<<< ours
<<<<<<< ours
    parser.add_argument("--max-evals", type=int, default=500, help="Iteraciones máximas del optimizador por subred")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base de RNG")
    parser.add_argument("--freeze-primes", type=int, nargs="*", help="Lista de primos a congelar (r_p=d_p=0)")
    parser.add_argument("--seed-sweep", type=int, default=10, help="Cantidad de semillas a evaluar")
    parser.add_argument(
        "--ablation",
        type=str,
        help="Subconjuntos de primos (formato: 2,3,5|2,3,5,7) para comparar",
    )
    parser.add_argument("--anchor-weight", type=float, help="Override rápido de w_anchor")
    parser.add_argument(
        "--bounds",
        nargs="*",
        help="Overrides de límites (ej: ratios=-0.2,0.2 deltas=-0.3,0.3 f0=18,22)",
    )
    parser.add_argument("--huber-delta", type=float, default=0.02, help="Delta para la pérdida Huber")
=======
    parser.add_argument("--max-evals", type=int, default=500, help="Evaluaciones máximas por subred")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de RNG")
>>>>>>> theirs
=======
    parser.add_argument("--max-evals", type=int, default=500, help="Evaluaciones máximas por subred")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de RNG")
>>>>>>> theirs
    return parser


def run_from_args(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = MaterialConfig.from_file(args.config)
    dataset = TargetDataset.from_file(args.targets)
    weights = load_loss_weights(args.weights)

<<<<<<< ours
<<<<<<< ours
    bounds_override = _parse_bounds_override(args.bounds)
    ablations = _parse_ablation_sets(args.ablation)
    freeze_primes = args.freeze_primes or None

    if args.anchor_weight is not None:
        weights.w_anchor = float(args.anchor_weight)

=======
>>>>>>> theirs
=======
>>>>>>> theirs
    engine = SimulationEngine(
        config=config,
        dataset=dataset,
        weights=weights,
        max_evals=args.max_evals,
        seed=args.seed,
<<<<<<< ours
<<<<<<< ours
        freeze_primes=freeze_primes,
        ablation_sets=ablations,
        seed_sweep=args.seed_sweep,
        bounds_override=bounds_override,
        huber_delta=args.huber_delta,
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    )
    bundle = engine.run()
    bundle.write(args.outdir, args.config, args.targets, args.max_evals, args.seed)


def main() -> None:
    run_from_args()


<<<<<<< ours
<<<<<<< ours
def _parse_bounds_override(entries: Optional[list[str]]) -> Optional[dict[str, Tuple[float, float]]]:
    if not entries:
        return None
    data: Dict[str, Tuple[float, float]] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        key, raw_values = entry.split("=", 1)
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if len(values) != 2:
            continue
        try:
            pair = [float(values[0]), float(values[1])]
        except ValueError:
            continue
        key_name = key.strip().lower()
        if key_name in {"ratios", "ratio", "ratios_bounds"}:
            data["ratios_bounds"] = tuple(pair)  # type: ignore[assignment]
        elif key_name in {"deltas", "delta", "deltas_bounds"}:
            data["deltas_bounds"] = tuple(pair)  # type: ignore[assignment]
        elif key_name in {"f0", "f0_bounds"}:
            data["f0_bounds"] = tuple(pair)  # type: ignore[assignment]
    return data or None


def _parse_ablation_sets(spec: Optional[str]) -> Optional[List[Sequence[int]]]:
    if not spec:
        return None
    subsets: List[List[int]] = []
    for block in spec.split("|"):
        values = [item.strip() for item in block.split(",") if item.strip()]
        if not values:
            continue
        try:
            subsets.append([int(value) for value in values])
        except ValueError:
            continue
    return subsets or None


if __name__ == "__main__":
    main()
=======
if __name__ == "__main__":
    main()

>>>>>>> theirs
=======
if __name__ == "__main__":
    main()

>>>>>>> theirs
