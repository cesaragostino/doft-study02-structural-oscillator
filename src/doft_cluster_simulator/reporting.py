"""Reporting utilities for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
from datetime import datetime

from .data import LossWeights, MaterialConfig, SubnetParameters, TargetDataset
from .loss import LossBreakdown
from .results import SubnetSimulation


@dataclass
class SubnetReport:
    name: str
    parameters: SubnetParameters
    loss: LossBreakdown
    e_sim: List[float]
    q_sim: Optional[float]
    residual_sim: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "L": self.parameters.L,
            "f0": self.parameters.f0,
            "ratios": self.parameters.ratios,
            "delta": self.parameters.delta,
            "layer_assignment": self.parameters.layer_assignment,
            "delta_T": getattr(self.parameters, "delta_T", {}),
            "delta_space": getattr(self.parameters, "delta_space", {}),
            "lambda_band": getattr(self.parameters, "lambda_band", None),
            "lambda_geo": getattr(self.parameters, "lambda_geo", None),
            "loss": self.loss.as_dict(),
            "e_sim": self.e_sim,
            "q_sim": self.q_sim,
            "residual_sim": self.residual_sim,
        }


@dataclass
class ContrastReport:
    pair: str
    target: float
    simulated: float
    loss: float


@dataclass
class ReportBundle:
    material: str
    weights: LossWeights
    subnets: Dict[str, SubnetReport]
    contrasts: List[ContrastReport]
    total_loss: float
    timestamp: str

    def write(
        self,
        out_dir: Path,
        config_path: Optional[Path],
        targets_path: Optional[Path],
        max_evals: int,
        seed: int,
        extras: Optional[dict[str, object]] = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_best_params(out_dir)
        self._write_csv(out_dir)
        self._write_report_md(out_dir)
        self._write_manifest(out_dir, config_path, targets_path, max_evals, seed, extras)

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
                    "delta_T": getattr(report.parameters, "delta_T", None),
                    "delta_space": getattr(report.parameters, "delta_space", None),
                    "delta_P": getattr(report.parameters, "delta_P", None),
                    "lambda_band": getattr(report.parameters, "lambda_band", None),
                    "lambda_geo": getattr(report.parameters, "lambda_geo", None),
                    "lambda_pressure_band": getattr(report.parameters, "lambda_pressure_band", None),
                    "lambda_pressure_geo": getattr(report.parameters, "lambda_pressure_geo", None),
                }
                for subnet, report in self.subnets.items()
            },
        }
        (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    def _write_csv(self, out_dir: Path) -> None:
        csv_path = out_dir / "simulation_results.csv"
        fieldnames = [
            "material",
            "subnet",
            "L",
            "f0",
            "e2",
            "e3",
            "e5",
            "e7",
            "q_sim",
            "residual_sim",
            "delta_T_2",
            "delta_T_3",
            "delta_T_5",
            "delta_T_7",
            "delta_space_2",
            "delta_space_3",
            "delta_space_5",
            "delta_space_7",
            "delta_P_2",
            "delta_P_3",
            "delta_P_5",
            "delta_P_7",
            "lambda_band",
            "lambda_pressure_band",
            "loss_total",
            "loss_e",
            "loss_q",
            "loss_residual",
            "loss_anchor",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
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
                    "delta_T_2": report.parameters.delta_T.get("2"),
                    "delta_T_3": report.parameters.delta_T.get("3"),
                    "delta_T_5": report.parameters.delta_T.get("5"),
                    "delta_T_7": report.parameters.delta_T.get("7"),
                    "delta_space_2": report.parameters.delta_space.get("2"),
                    "delta_space_3": report.parameters.delta_space.get("3"),
                    "delta_space_5": report.parameters.delta_space.get("5"),
                    "delta_space_7": report.parameters.delta_space.get("7"),
                    "lambda_band": getattr(report.parameters, "lambda_band", None),
                    "delta_P_2": report.parameters.delta_P.get("2"),
                    "delta_P_3": report.parameters.delta_P.get("3"),
                    "delta_P_5": report.parameters.delta_P.get("5"),
                    "delta_P_7": report.parameters.delta_P.get("7"),
                    "lambda_pressure_band": getattr(report.parameters, "lambda_pressure_band", None),
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
        (out_dir / "report.md").write_text("\n".join(lines))

    def _write_manifest(
        self,
        out_dir: Path,
        config_path: Optional[Path],
        targets_path: Optional[Path],
        max_evals: int,
        seed: int,
        extras: Optional[dict[str, object]],
    ) -> None:
        manifest = {
            "version": "0.9",
            "generated_at": self.timestamp,
            "material": self.material,
            "config_path": str(config_path) if config_path else None,
            "targets_path": str(targets_path) if targets_path else None,
            "max_evals": max_evals,
            "seed": seed,
            "weights": {
                "w_e": self.weights.w_e,
                "w_q": self.weights.w_q,
                "w_r": self.weights.w_r,
                "w_c": self.weights.w_c,
                "w_anchor": self.weights.w_anchor,
            },
            "total_loss": self.total_loss,
        }
        if extras:
            manifest["extras"] = extras
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


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
