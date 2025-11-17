"""Reporting utilities for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
<<<<<<< ours
<<<<<<< ours
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import json
from datetime import UTC, datetime
import math
import subprocess
import uuid

from .data import LossWeights, MaterialConfig, ParameterBounds, SubnetTarget, TargetDataset
from .loss import LossBreakdown
from .results import SimulationRun, SubnetSimulation
=======
=======
>>>>>>> theirs
from typing import Dict, List, Optional
import csv
import json
from datetime import datetime

from .data import LossWeights, MaterialConfig, SubnetParameters, TargetDataset
from .loss import LossBreakdown
from .results import SubnetSimulation
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs


@dataclass
class SubnetReport:
    name: str
<<<<<<< ours
<<<<<<< ours
    parameters: SubnetSimulation
=======
    parameters: SubnetParameters
>>>>>>> theirs
=======
    parameters: SubnetParameters
>>>>>>> theirs
    loss: LossBreakdown
    e_sim: List[float]
    q_sim: Optional[float]
    residual_sim: float
<<<<<<< ours
<<<<<<< ours
    e_abs_errors: List[Optional[float]]
    e_abs_mean: Optional[float]
    q_error: Optional[float]
    residual_error: Optional[float]
    f0_anchor_error: Optional[float]
    q_gated: bool
    q_gated: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "L": self.parameters.parameters.L,
            "f0": self.parameters.parameters.f0,
            "ratios": self.parameters.parameters.ratios,
            "delta": self.parameters.parameters.delta,
            "layer_assignment": self.parameters.parameters.layer_assignment,
=======
=======
>>>>>>> theirs

    def to_dict(self) -> Dict[str, object]:
        return {
            "L": self.parameters.L,
            "f0": self.parameters.f0,
            "ratios": self.parameters.ratios,
            "delta": self.parameters.delta,
            "layer_assignment": self.parameters.layer_assignment,
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
            "loss": self.loss.as_dict(),
            "e_sim": self.e_sim,
            "q_sim": self.q_sim,
            "residual_sim": self.residual_sim,
<<<<<<< ours
<<<<<<< ours
            "e_abs_errors": self.e_abs_errors,
            "e_abs_mean": self.e_abs_mean,
            "q_error": self.q_error,
            "residual_error": self.residual_error,
            "f0_anchor_error": self.f0_anchor_error,
            "q_gated": self.q_gated,
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        }


@dataclass
class ContrastReport:
    pair: str
    target: float
    simulated: float
    loss: float

<<<<<<< ours
<<<<<<< ours
    @property
    def error(self) -> float:
        return abs(self.simulated - self.target)


@dataclass
class RunReport:
    label: str
    seed: int
    primes: Tuple[int, ...]
    freeze_primes: Tuple[int, ...]
    subnets: Dict[str, SubnetReport]
    contrasts: List[ContrastReport]
    total_loss: float
    loss_components: Dict[str, float]
    metric_summary: Dict[str, float]


@dataclass
class AggregateStats:
    f0: Dict[str, Tuple[float, float]]
    e_mean: Dict[str, Tuple[float, float]]
    q: Dict[str, Tuple[float, float]]
    residual: Dict[str, Tuple[float, float]]
    contrast: Dict[str, Tuple[float, float]]


@dataclass
class AblationStats:
    prime_signature: str
    mean_total_loss: float
    mean_contrast_error: float

=======
>>>>>>> theirs
=======
>>>>>>> theirs

@dataclass
class ReportBundle:
    material: str
    weights: LossWeights
<<<<<<< ours
<<<<<<< ours
    runs: List[RunReport]
    aggregates: AggregateStats
    ablations: List[AblationStats]
    best_run_label: str
    timestamp: str
    run_id: str
    schema_version: str
    seed_list: List[int]
    seed_sweep: int
    bounds: ParameterBounds
    targets: Dict[str, SubnetTarget]
    run_id: str
    schema_version: str
    seed_list: List[int]
    seed_sweep: int
    bounds: ParameterBounds

    def write(
        self,
        out_dir: Path,
        config_path: Optional[Path],
        targets_path: Optional[Path],
        max_evals: int,
        seed: int,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        best_run = self._find_best_run()
        self._write_best_params(out_dir, best_run)
        self._write_csv(out_dir)
        self._write_report_md(out_dir, best_run)
        self._write_manifest(out_dir, config_path, targets_path, max_evals, seed)

    def _find_best_run(self) -> RunReport:
        return min(self.runs, key=lambda run: run.total_loss)

    def _write_best_params(self, out_dir: Path, run: RunReport) -> None:
        best_params = {
            "material": self.material,
            "run_label": run.label,
            "params": {
                subnet: report.to_dict()
                for subnet, report in run.subnets.items()
=======
=======
>>>>>>> theirs
    subnets: Dict[str, SubnetReport]
    contrasts: List[ContrastReport]
    total_loss: float
    timestamp: str

    def write(self, out_dir: Path, config_path: Optional[Path], targets_path: Optional[Path], max_evals: int, seed: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_best_params(out_dir)
        self._write_csv(out_dir)
        self._write_report_md(out_dir)
        self._write_manifest(out_dir, config_path, targets_path, max_evals, seed)

    def _write_best_params(self, out_dir: Path) -> None:
        best_params = {
            "material": self.material,
            "params": {
                subnet: {
                    "L": report.parameters.L,
                    "f0": report.parameters.f0,
                    "ratios": report.parameters.ratios,
                    "delta": report.parameters.delta,
                    "layer_assignment": report.parameters.layer_assignment,
                }
                for subnet, report in self.subnets.items()
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
            },
        }
        (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    def _write_csv(self, out_dir: Path) -> None:
        csv_path = out_dir / "simulation_results.csv"
        fieldnames = [
<<<<<<< ours
<<<<<<< ours
            "run_label",
            "seed",
            "primes",
            "freeze_primes",
=======
            "material",
>>>>>>> theirs
=======
            "material",
>>>>>>> theirs
            "subnet",
            "L",
            "f0",
            "e2",
            "e3",
            "e5",
            "e7",
<<<<<<< ours
<<<<<<< ours
            "e_mean_abs_error",
            "q_sim",
            "q_error",
            "residual_sim",
            "residual_error",
            "f0_anchor_error",
=======
            "q_sim",
            "residual_sim",
>>>>>>> theirs
=======
            "q_sim",
            "residual_sim",
>>>>>>> theirs
            "loss_total",
            "loss_e",
            "loss_q",
            "loss_residual",
            "loss_anchor",
<<<<<<< ours
<<<<<<< ours
            "loss_regularization",
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
<<<<<<< ours
<<<<<<< ours
            for run in self.runs:
                for subnet, report in run.subnets.items():
                    row = {
                        "run_label": run.label,
                        "seed": run.seed,
                        "primes": ",".join(str(p) for p in run.primes),
                        "freeze_primes": ",".join(str(p) for p in run.freeze_primes),
                        "subnet": subnet,
                        "L": report.parameters.parameters.L,
                        "f0": report.parameters.parameters.f0,
                        "e2": report.e_sim[0],
                        "e3": report.e_sim[1],
                        "e5": report.e_sim[2],
                        "e7": report.e_sim[3],
                        "e_mean_abs_error": report.e_abs_mean,
                        "q_sim": report.q_sim,
                        "q_error": report.q_error,
                        "residual_sim": report.residual_sim,
                        "residual_error": report.residual_error,
                        "f0_anchor_error": report.f0_anchor_error,
                        "loss_total": report.loss.total,
                        "loss_e": report.loss.e_loss,
                        "loss_q": report.loss.q_loss,
                        "loss_residual": report.loss.residual_loss,
                        "loss_anchor": report.loss.anchor_loss,
                        "loss_regularization": report.loss.regularization_loss,
                    }
                    writer.writerow(row)

    def _write_report_md(self, out_dir: Path, best_run: RunReport) -> None:
        lines: List[str] = []
        lines.append(f"# DOFT Cluster Simulator Results — {self.material}")
        lines.append("")
        lines.append(f"Fecha de ejecución: {self.timestamp}")
        lines.append("")
        lines.append(f"Schema version: {self.schema_version} · Run ID: {self.run_id}")
        lines.append("")
        lines.append(f"Seeds ({self.seed_sweep}): {', '.join(str(seed) for seed in self.seed_list)}")
        lines.append("")
        lines.append(
            f"Bounds → ratios {self.bounds.ratios}, deltas {self.bounds.deltas}, f0 {self.bounds.f0}"
        )
        lines.append("")
        lines.append(
            f"Pesos → w_e={self.weights.w_e}, w_q={self.weights.w_q}, w_r={self.weights.w_r}, "
            f"w_c={self.weights.w_c}, w_anchor={self.weights.w_anchor}, λ={self.weights.lambda_reg}"
        )
        lines.append("")
        lines.append(f"**Mejor corrida:** `{best_run.label}` con pérdida total {best_run.total_loss:.6f}")
        lines.append("")
        lines.append("## Runs")
        lines.append("")
        lines.append("| Run | Seed | Primes | Freeze | L_total | L_e | L_q | L_r | L_c | L_anchor | L_reg |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for run in self.runs:
            components = run.loss_components
            lines.append(
                f"| `{run.label}` | {run.seed} | {','.join(str(p) for p in run.primes)} | "
                f"{','.join(str(p) for p in run.freeze_primes) or '—'} | {run.total_loss:.6f} | "
                f"{components['e']:.6f} | {components['q']:.6f} | {components['r']:.6f} | "
                f"{components['contrast']:.6f} | {components['anchor']:.6f} | {components['regularization']:.6f} |"
            )
        lines.append("")

        lines.append("## Métricas agregadas (media ± desviación)")
        lines.append("")
        lines.append("| Subred | f0 | |e| | q | residual |")
        lines.append("| --- | --- | --- | --- | --- |")
        for subnet in sorted(self.aggregates.f0.keys()):
            f0_mean, f0_std = self.aggregates.f0[subnet]
            e_mean, e_std = self.aggregates.e_mean.get(subnet, (0.0, 0.0))
            q_mean, q_std = self.aggregates.q.get(subnet, (0.0, 0.0))
            r_mean, r_std = self.aggregates.residual.get(subnet, (0.0, 0.0))
            lines.append(
                f"| {subnet} | {f0_mean:.4f} ± {f0_std:.4f} | {e_mean:.4f} ± {e_std:.4f} | "
                f"{q_mean:.4f} ± {q_std:.4f} | {r_mean:.5f} ± {r_std:.5f} |"
            )
        lines.append("")

        lines.append("## Contrastes")
        lines.append("")
        lines.append("| Par | error medio ± std |")
        lines.append("| --- | --- |")
        for pair, (mean_val, std_val) in self.aggregates.contrast.items():
            lines.append(f"| {pair} | {mean_val:.5f} ± {std_val:.5f} |")
        lines.append("")

        if self.ablations:
            lines.append("## Ablaciones por primos")
            lines.append("")
            lines.append("| Primos | pérdida media | error contraste medio |")
            lines.append("| --- | --- | --- |")
            for entry in self.ablations:
                lines.append(
                    f"| {entry.prime_signature} | {entry.mean_total_loss:.6f} | {entry.mean_contrast_error:.5f} |"
                )
            lines.append("")

        lines.append("## Mejor corrida — detalle por subred")
        lines.append("")
        lines.append("| Subred | e_sim | e_exp | q_sim | q_exp | residual_sim | residual_exp | q_gated |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for subnet, report in best_run.subnets.items():
            target_key = f"{self.material}_{subnet}"
            target = self.targets.get(target_key)
            e_exp = target.e_exp if target else None
            q_exp = target.q_exp if target else None
            residual_exp = target.residual_exp if target else None
            lines.append(
                f"| {subnet} | { _format_list(report.e_sim) } | { _format_list(e_exp) } | "
                f"{_format_optional(report.q_sim)} | {_format_optional(q_exp)} | "
                f"{_format_optional(report.residual_sim)} | {_format_optional(residual_exp)} | {report.q_gated} |"
            )
        lines.append("")

=======
=======
>>>>>>> theirs
            for subnet, report in self.subnets.items():
                row = {
                    "material": self.material,
                    "subnet": subnet,
                    "L": report.parameters.L,
                    "f0": report.parameters.f0,
                    "e2": report.e_sim[0],
                    "e3": report.e_sim[1],
                    "e5": report.e_sim[2],
                    "e7": report.e_sim[3],
                    "q_sim": report.q_sim,
                    "residual_sim": report.residual_sim,
                    "loss_total": report.loss.total,
                    "loss_e": report.loss.e_loss,
                    "loss_q": report.loss.q_loss,
                    "loss_residual": report.loss.residual_loss,
                    "loss_anchor": report.loss.anchor_loss,
                }
                writer.writerow(row)

    def _write_report_md(self, out_dir: Path) -> None:
        lines = [f"# DOFT Cluster Simulator Results — {self.material}", ""]
        lines.append(f"Fecha de ejecución: {self.timestamp}")
        lines.append("")
        lines.append("## Ajuste por subred")
        lines.append("")
        lines.append("| Subred | L | f0 | e2 | e3 | e5 | e7 | q_sim | residual_sim | pérdida |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for subnet, report in self.subnets.items():
            q_display = f"{report.q_sim:.4f}" if report.q_sim is not None else "N/A"
            lines.append(
                f"| {subnet} | {report.parameters.L} | {report.parameters.f0:.4f} | "
                f"{report.e_sim[0]:.4f} | {report.e_sim[1]:.4f} | {report.e_sim[2]:.4f} | {report.e_sim[3]:.4f} | "
                f"{q_display} | {report.residual_sim:.6f} | {report.loss.total:.6f} |"
            )
        lines.append("")
        if self.contrasts:
            lines.append("## Contrastes")
            lines.append("")
            lines.append("| Par | objetivo | simulado | pérdida |")
            lines.append("| --- | --- | --- | --- |")
            for contrast in self.contrasts:
                lines.append(
                    f"| {contrast.pair} | {contrast.target:.5f} | {contrast.simulated:.5f} | {contrast.loss:.6f} |"
                )
            lines.append("")
        lines.append(f"**Pérdida total:** {self.total_loss:.6f}")
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
        (out_dir / "report.md").write_text("\n".join(lines))

    def _write_manifest(
        self,
        out_dir: Path,
        config_path: Optional[Path],
        targets_path: Optional[Path],
        max_evals: int,
        seed: int,
    ) -> None:
<<<<<<< ours
<<<<<<< ours
        best_run = self._find_best_run()
        manifest = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
=======
        manifest = {
            "version": "0.9",
>>>>>>> theirs
=======
        manifest = {
            "version": "0.9",
>>>>>>> theirs
            "generated_at": self.timestamp,
            "material": self.material,
            "config_path": str(config_path) if config_path else None,
            "targets_path": str(targets_path) if targets_path else None,
            "max_evals": max_evals,
            "seed": seed,
<<<<<<< ours
<<<<<<< ours
            "seed_sweep": self.seed_sweep,
            "seeds": self.seed_list,
            "run_count": len(self.runs),
            "best_run": self.best_run_label,
=======
>>>>>>> theirs
=======
>>>>>>> theirs
            "weights": {
                "w_e": self.weights.w_e,
                "w_q": self.weights.w_q,
                "w_r": self.weights.w_r,
                "w_c": self.weights.w_c,
                "w_anchor": self.weights.w_anchor,
<<<<<<< ours
<<<<<<< ours
                "lambda_reg": self.weights.lambda_reg,
            },
            "bounds": {
                "ratios": list(self.bounds.ratios),
                "deltas": list(self.bounds.deltas),
                "f0": list(self.bounds.f0),
            },
            "loss_components": best_run.loss_components,
            "metric_summary": best_run.metric_summary,
            "commit": _get_commit_sha(),
=======
            },
            "total_loss": self.total_loss,
>>>>>>> theirs
=======
            },
            "total_loss": self.total_loss,
>>>>>>> theirs
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


<<<<<<< ours
<<<<<<< ours
def create_report_bundle(
    config: MaterialConfig,
    weights: LossWeights,
    runs: List[SimulationRun],
    contrast_targets: List[ContrastTarget],
    dataset: TargetDataset,
    seed_list: List[int],
    seed_sweep: int,
    run_id: str,
    schema_version: str,
    bounds: ParameterBounds,
) -> ReportBundle:
    run_reports: List[RunReport] = []
    for run in runs:
        sub_reports: Dict[str, SubnetReport] = {}
        for subnet, result in run.subnet_results.items():
            target_key = f"{config.material}_{subnet}"
            target = dataset.subnets.get(target_key)
            anchor_value = None
            anchor_info = config.anchors.get(subnet)
            if anchor_info is not None:
                anchor_value = anchor_info.get("f0") or anchor_info.get("X")
            sub_reports[subnet] = _build_subnet_report(subnet, result, target, anchor_value)

        contrasts: List[ContrastReport] = []
        loss_components = _initial_loss_components()
        for report in sub_reports.values():
            loss_components["e"] += report.loss.e_loss
            loss_components["q"] += report.loss.q_loss
            loss_components["r"] += report.loss.residual_loss
            loss_components["anchor"] += report.loss.anchor_loss
            loss_components["regularization"] += report.loss.regularization_loss
        total_loss = run.base_loss
        for contrast in contrast_targets:
            report_a = sub_reports.get(_strip_material_prefix(contrast.subnet_a, config.material))
            report_b = sub_reports.get(_strip_material_prefix(contrast.subnet_b, config.material))
            if report_a is None or report_b is None or contrast.value is None:
                if report_a is None or report_b is None:
                    raise ValueError(
                        f"Contrast '{contrast.label or contrast.subnet_a}' references missing subnets"
                    )
            simulated = compute_contrast_value(report_a, report_b)
            loss = weights.w_c * abs(simulated - contrast.value)
            contrasts.append(
                ContrastReport(
                    pair=contrast.label or f"{report_a.name}_vs_{report_b.name}",
                    target=contrast.value,
                    simulated=simulated,
                    loss=loss,
                )
            )
            total_loss += loss
            loss_components["contrast"] += loss

        metric_summary = {
            "e_abs_mean": _mean_or_zero(report.e_abs_mean for report in sub_reports.values()),
            "q_abs_mean": _mean_or_zero(report.q_error for report in sub_reports.values()),
            "residual_abs_mean": _mean_or_zero(report.residual_error for report in sub_reports.values()),
            "contrast_abs_mean": _mean_or_zero(contrast.error for contrast in contrasts),
        }

        run_reports.append(
            RunReport(
                label=run.label,
                seed=run.seed,
                primes=run.primes,
                freeze_primes=run.freeze_primes,
                subnets=sub_reports,
                contrasts=contrasts,
                total_loss=total_loss,
                loss_components=loss_components,
                metric_summary=metric_summary,
            )
        )

    aggregates = _compute_aggregate_stats(run_reports)
    ablations = _compute_ablation_stats(run_reports)
    best_run_label = min(run_reports, key=lambda r: r.total_loss).label
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    return ReportBundle(
        material=config.material,
        weights=weights,
        runs=run_reports,
        aggregates=aggregates,
        ablations=ablations,
        best_run_label=best_run_label,
        timestamp=timestamp,
        run_id=run_id,
        schema_version=schema_version,
        seed_list=seed_list,
        seed_sweep=seed_sweep,
        bounds=bounds,
        targets=dict(dataset.subnets),
    )


# ---------------------------------------------------------------------------
# Helpers


def _build_subnet_report(
    name: str,
    simulation: SubnetSimulation,
    target: Optional[SubnetTarget],
    anchor_value: Optional[float],
) -> SubnetReport:
    e_abs_errors: List[Optional[float]] = []
    e_abs_mean: Optional[float] = None
    q_error: Optional[float] = None
    residual_error: Optional[float] = None
    f0_anchor_error: Optional[float] = None

    if target and target.e_exp is not None:
        diffs: List[float] = []
        for idx, exp_value in enumerate(target.e_exp):
            if exp_value is None:
                e_abs_errors.append(None)
                continue
            diff = abs(simulation.simulation_result.e_sim[idx] - exp_value)
            e_abs_errors.append(diff)
            diffs.append(diff)
        if diffs:
            e_abs_mean = sum(diffs) / len(diffs)
    else:
        e_abs_errors = [None, None, None, None]

    q_gated = bool(target and not target.use_q)
    if target and target.use_q and target.q_exp is not None and simulation.simulation_result.q_sim is not None:
        q_error = abs(simulation.simulation_result.q_sim - target.q_exp)

    if target and target.residual_exp is not None:
        residual_error = abs(simulation.simulation_result.residual_sim - target.residual_exp)

    if anchor_value is not None:
        f0_anchor_error = abs(simulation.parameters.f0 - anchor_value)

    return SubnetReport(
        name=name,
        parameters=simulation,
        loss=simulation.loss,
        e_sim=simulation.simulation_result.e_sim,
        q_sim=simulation.simulation_result.q_sim,
        residual_sim=simulation.simulation_result.residual_sim,
        e_abs_errors=e_abs_errors,
        e_abs_mean=e_abs_mean,
        q_error=q_error,
        residual_error=residual_error,
        f0_anchor_error=f0_anchor_error,
        q_gated=q_gated,
    )


def compute_contrast_value(a: SubnetReport, b: SubnetReport) -> float:
    scale_a = a.parameters.parameters.f0 + sum(a.e_sim) / max(len(a.e_sim), 1)
    scale_b = b.parameters.parameters.f0 + sum(b.e_sim) / max(len(b.e_sim), 1)
    return scale_a / max(scale_b, 1e-6)


def _strip_material_prefix(name: str, material: str) -> str:
    prefix = f"{material}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _compute_aggregate_stats(runs: List[RunReport]) -> AggregateStats:
    f0: Dict[str, List[float]] = {}
    e_mean: Dict[str, List[float]] = {}
    q_errors: Dict[str, List[float]] = {}
    residual_errors: Dict[str, List[float]] = {}
    contrast_errors: Dict[str, List[float]] = {}

    for run in runs:
        for subnet, report in run.subnets.items():
            f0.setdefault(subnet, []).append(report.parameters.parameters.f0)
            if report.e_abs_mean is not None:
                e_mean.setdefault(subnet, []).append(report.e_abs_mean)
            if report.q_error is not None:
                q_errors.setdefault(subnet, []).append(report.q_error)
            if report.residual_error is not None:
                residual_errors.setdefault(subnet, []).append(report.residual_error)
        for contrast in run.contrasts:
            contrast_errors.setdefault(contrast.pair, []).append(contrast.error)

    return AggregateStats(
        f0={key: _mean_std(values) for key, values in f0.items()},
        e_mean={key: _mean_std(values) for key, values in e_mean.items()},
        q={key: _mean_std(values) for key, values in q_errors.items()},
        residual={key: _mean_std(values) for key, values in residual_errors.items()},
        contrast={key: _mean_std(values) for key, values in contrast_errors.items()},
    )


def _compute_ablation_stats(runs: List[RunReport]) -> List[AblationStats]:
    groups: Dict[Tuple[int, ...], List[RunReport]] = {}
    for run in runs:
        groups.setdefault(run.primes, []).append(run)
    stats: List[AblationStats] = []
    for primes, entries in groups.items():
        mean_loss = sum(run.total_loss for run in entries) / len(entries)
        contrast_values: List[float] = []
        for run in entries:
            contrast_values.extend(contrast.error for contrast in run.contrasts)
        mean_contrast = sum(contrast_values) / len(contrast_values) if contrast_values else 0.0
        stats.append(
            AblationStats(
                prime_signature=",".join(str(p) for p in primes),
                mean_total_loss=mean_loss,
                mean_contrast_error=mean_contrast,
            )
        )
    return stats


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    data = list(values)
    if not data:
        return (0.0, 0.0)
    mean = sum(data) / len(data)
    if len(data) == 1:
        return (mean, 0.0)
    variance = sum((value - mean) ** 2 for value in data) / (len(data) - 1)
    return (mean, math.sqrt(max(variance, 0.0)))


def _mean_or_zero(values: Iterable[Optional[float]]) -> float:
    data = [value for value in values if value is not None]
    return sum(data) / len(data) if data else 0.0


def _initial_loss_components() -> Dict[str, float]:
    return {"e": 0.0, "q": 0.0, "r": 0.0, "anchor": 0.0, "regularization": 0.0, "contrast": 0.0}


def _format_list(values: Optional[List[Optional[float]]]) -> str:
    if not values:
        return "—"
    return ", ".join("—" if value is None else f"{value:.3f}" for value in values)


def _format_optional(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.5f}" if abs(value) < 1 else f"{value:.3f}"


def _get_commit_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None
=======
=======
>>>>>>> theirs
def compute_contrast_value(a: SubnetReport, b: SubnetReport) -> float:
    scale_a = a.parameters.f0 + sum(a.e_sim) / max(len(a.e_sim), 1)
    scale_b = b.parameters.f0 + sum(b.e_sim) / max(len(b.e_sim), 1)
    return scale_a / max(scale_b, 1e-6)


def create_report_bundle(
    config: MaterialConfig,
    dataset: TargetDataset,
    weights: LossWeights,
    subnet_results: Dict[str, SubnetSimulation],
) -> ReportBundle:
    sub_reports: Dict[str, SubnetReport] = {}
    total_loss = 0.0
    for subnet, result in subnet_results.items():
        report = SubnetReport(
            name=subnet,
            parameters=result.parameters,
            loss=result.loss,
            e_sim=result.simulation_result.e_sim,
            q_sim=result.simulation_result.q_sim,
            residual_sim=result.simulation_result.residual_sim,
        )
        sub_reports[subnet] = report
        total_loss += result.loss.total

    contrast_reports: List[ContrastReport] = []
    for contrast in dataset.contrasts:
        a_name = contrast.subnet_a.split(f"{config.material}_")[-1]
        b_name = contrast.subnet_b.split(f"{config.material}_")[-1]
        report_a = sub_reports.get(a_name)
        report_b = sub_reports.get(b_name)
        if report_a is None or report_b is None:
            continue
        simulated = compute_contrast_value(report_a, report_b)
        loss = ((simulated - contrast.value) ** 2) * weights.w_c
        contrast_reports.append(ContrastReport(pair=f"{a_name}_vs_{b_name}", target=contrast.value, simulated=simulated, loss=loss))
        total_loss += loss

    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return ReportBundle(
        material=config.material,
        weights=weights,
        subnets=sub_reports,
        contrasts=contrast_reports,
        total_loss=total_loss,
        timestamp=timestamp,
    )

<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
