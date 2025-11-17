from __future__ import annotations

import json
from pathlib import Path

from scripts.doft_cluster_simulator.data import LossWeights, MaterialConfig, SubnetParameters, SubnetTarget, TargetDataset
from scripts.doft_cluster_simulator.engine import SimulationEngine
from scripts.doft_cluster_simulator.loss import compute_subnet_loss
from scripts.doft_cluster_simulator.model import SimulationResult


def test_target_dataset_handles_missing_values(tmp_path: Path) -> None:
    targets = {
        "MgB2_sigma": {"e_exp": [1.0, 0.0, 0.0, 0.0], "q_exp": None, "residual_exp": -0.01},
        "MgB2_pi": {"e_exp": [1.2, 0.7, 0.2, 0.4], "q_exp": 6.0, "residual_exp": -0.02},
        "MgB2_sigma_vs_pi": {"C_AB_exp": 1.5},
    }
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(targets))

    dataset = TargetDataset.from_file(path)
    assert dataset.subnets["MgB2_sigma"].q_exp is None
    assert dataset.contrasts[0].subnet_b == "MgB2_pi"


def test_loss_gates_missing_terms() -> None:
    params = SubnetParameters(L=2, f0=1.5, ratios={"r2": 1.0, "r3": 0.5, "r5": 0.2, "r7": 0.1}, delta={"d2": 0.0, "d3": 0.0, "d5": 0.0, "d7": 0.0}, layer_assignment=[1, 1, 2, 2])
    simulation = SimulationResult(e_sim=[1.0, 0.8, 0.2, 0.4], q_sim=None, residual_sim=-0.015, layer_factors=[1.0, 1.0, 1.18, 1.18])
    target = SubnetTarget(e_exp=[1.0, 0.7, None, None], q_exp=None, residual_exp=None)
    weights = LossWeights(w_e=1.0, w_q=2.0, w_r=3.0)

    breakdown = compute_subnet_loss(target, params, simulation, weights, anchor_value=None)
    assert breakdown.q_loss == 0.0
    assert breakdown.residual_loss == 0.0
    assert breakdown.e_loss > 0.0


def test_engine_runs_end_to_end(tmp_path: Path) -> None:
    config_data = {"material": "MgB2", "subnets": ["sigma", "pi"], "anchors": {"sigma": {"X": 1.5}, "pi": {"X": 1.6}}}
    targets = {
        "MgB2_sigma": {"e_exp": [1.0, 0.0, 0.0, 0.0], "q_exp": None, "residual_exp": -0.01},
        "MgB2_pi": {"e_exp": [1.2, 0.7, 0.2, 0.4], "q_exp": 6.0, "residual_exp": -0.02},
        "MgB2_sigma_vs_pi": {"C_AB_exp": 1.5},
    }

    config_path = tmp_path / "config.json"
    targets_path = tmp_path / "targets.json"
    config_path.write_text(json.dumps(config_data))
    targets_path.write_text(json.dumps(targets))

    config = MaterialConfig.from_file(config_path)
    dataset = TargetDataset.from_file(targets_path)
    weights = LossWeights()

    engine = SimulationEngine(config=config, dataset=dataset, weights=weights, max_evals=10, seed=123)
    bundle = engine.run()
    out_dir = tmp_path / "out"
    bundle.write(out_dir, config_path, targets_path, max_evals=10, seed=123)

    assert (out_dir / "best_params.json").exists()
    assert (out_dir / "simulation_results.csv").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "manifest.json").exists()

