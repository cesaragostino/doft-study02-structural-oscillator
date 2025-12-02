"""Small helper to visualise how delta_T and delta_space shift the residual loss.

Run with:
    PYTHONPATH=. python src/tools/demo_delta_T_effect.py
"""

from __future__ import annotations

from src.doft_cluster_simulator.data import LossWeights, SubnetParameters, SubnetTarget
from src.doft_cluster_simulator.loss import compute_subnet_loss
from src.doft_cluster_simulator.model import SimulationResult
from src.doft_cluster_simulator.data import PRIMES


def _prime_dict(value: float) -> dict:
    return {str(p): float(value) for p in PRIMES}


def main() -> None:
    # Minimal setup: no xi shift, only delta_T/delta_space vary
    params = SubnetParameters(
        L=1,
        f0=1.0,
        ratios={"r2": 0.0, "r3": 0.0, "r5": 0.0, "r7": 0.0},
        delta={"d2": 0.0, "d3": 0.0, "d5": 0.0, "d7": 0.0},
        layer_assignment=[1, 1, 1, 1],
        delta_T=_prime_dict(0.0),
        delta_space=_prime_dict(0.0),
    )
    simulation = SimulationResult(
        e_sim=[0.0, 0.0, 0.0, 0.0],
        q_sim=None,
        residual_sim=0.5,
        layer_factors=[1.0, 1.0, 1.0, 1.0],
    )
    target = SubnetTarget(e_exp=None, q_exp=None, residual_exp=0.0)
    weights = LossWeights(w_r=1.0)

    for delta_T_val, delta_space_val in ((0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (0.05, 0.05), (-0.05, 0.1)):
        params.delta_T = _prime_dict(delta_T_val)
        params.delta_space = _prime_dict(delta_space_val)
        loss = compute_subnet_loss(
            target=target,
            params=params,
            simulation=simulation,
            weights=weights,
            anchor_value=None,
            xi_value=None,
            xi_sign=1,
            xi_exp=None,
            k_skin=0.0,
            delta_T=params.delta_T,
            delta_space=params.delta_space,
        )
        adjusted_residual = simulation.residual_sim - delta_T_val - delta_space_val
        print(
            f"delta_T={delta_T_val:+.3f}, delta_space={delta_space_val:+.3f} -> "
            f"adjusted_residual={adjusted_residual:+.3f}, residual_loss={loss.residual_loss:.6f}"
        )


if __name__ == "__main__":
    main()
