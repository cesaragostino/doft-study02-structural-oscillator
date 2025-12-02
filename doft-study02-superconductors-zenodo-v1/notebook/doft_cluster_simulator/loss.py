"""Loss computation utilities for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .data import LossWeights, PRIMES, SubnetParameters, SubnetTarget
from .model import SimulationResult


@dataclass
class LossBreakdown:
    """Detailed loss information for a subnet."""

    total: float
    e_loss: float
    q_loss: float
    residual_loss: float
    anchor_loss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "e": self.e_loss,
            "q": self.q_loss,
            "residual": self.residual_loss,
            "anchor": self.anchor_loss,
        }


def _safe_len(values: Optional[List[Optional[float]]]) -> int:
    if values is None:
        return 0
    return sum(1 for value in values if value is not None)


def compute_subnet_loss(
    target: SubnetTarget,
    params: SubnetParameters,
    simulation: SimulationResult,
    weights: LossWeights,
    anchor_value: Optional[float],
    huber_delta: Optional[float] = None,
    xi_value: Optional[float] = None,
    xi_sign: int = 0,
    xi_exp: Optional[dict] = None,
    k_skin: float = 0.0,
    lambda_band: float = 1.0,
    lambda_geo: Optional[dict] = None,
    lambda_pressure_band: float = 1.0,
    lambda_pressure_geo: Optional[dict] = None,
) -> LossBreakdown:
    """Compute the weighted loss for a subnet."""

    def _loss_term(diff: float) -> float:
        if huber_delta is None or huber_delta <= 0:
            return diff * diff
        abs_diff = abs(diff)
        if abs_diff <= huber_delta:
            return 0.5 * abs_diff * abs_diff
        return huber_delta * (abs_diff - 0.5 * huber_delta)

    e_terms = []
    if target.e_exp is not None:
        for idx, value in enumerate(target.e_exp):
            if value is None:
                continue
            diff = simulation.e_sim[idx] - value
            e_terms.append(_loss_term(diff))
    e_loss = (sum(e_terms) / max(len(e_terms), 1)) * weights.w_e if e_terms else 0.0

    q_loss = 0.0
    if target.q_exp is not None and simulation.q_sim is not None:
        diff_q = simulation.q_sim - target.q_exp
        q_loss = _loss_term(diff_q) * weights.w_q

    residual_loss = 0.0
    if target.residual_exp is not None:
        residual_adjusted = simulation.residual_sim
        if xi_value is not None:
            residual_adjusted = residual_adjusted - (xi_sign or 0) * xi_value
            residual_adjusted = residual_adjusted - (xi_sign or 0) * (k_skin or 0.0) * xi_value
        if xi_exp:
            exp_shift = sum(float(v) for v in xi_exp.values())
            residual_adjusted = residual_adjusted - (xi_sign or 0) * exp_shift
        lambda_geo_vec = {str(p): float(lambda_geo.get(str(p), 1.0)) for p in PRIMES} if lambda_geo else {str(p): 1.0 for p in PRIMES}
        delta_T_vec = {str(k): float(v) for k, v in params.delta_T.items()} if params.delta_T else {}
        delta_space_vec = {str(k): float(v) for k, v in params.delta_space.items()} if params.delta_space else {}
        delta_P_vec = {str(k): float(v) for k, v in params.delta_P.items()} if getattr(params, "delta_P", None) else {}
        lambda_pressure_geo_vec = (
            {str(p): float(lambda_pressure_geo.get(str(p), 1.0)) for p in PRIMES} if lambda_pressure_geo else {str(p): 1.0 for p in PRIMES}
        )
        for prime in PRIMES:
            key = str(prime)
            beta = lambda_geo_vec.get(key, 1.0)
            lambda_total = float(lambda_band or 1.0) * lambda_geo_vec.get(key, 1.0)
            lambda_pressure_total = float(lambda_pressure_band or 1.0) * lambda_pressure_geo_vec.get(key, 1.0)
            residual_adjusted = residual_adjusted - beta * delta_T_vec.get(key, 0.0)
            residual_adjusted = residual_adjusted - lambda_total * delta_space_vec.get(key, 0.0)
            residual_adjusted = residual_adjusted - lambda_pressure_total * delta_P_vec.get(key, 0.0)
        diff_r = residual_adjusted - target.residual_exp
        residual_loss = _loss_term(diff_r) * weights.w_r

    anchor_loss = 0.0
    if anchor_value is not None:
        diff_a = params.f0 - anchor_value
        anchor_loss = _loss_term(diff_a) * weights.w_anchor

    total = e_loss + q_loss + residual_loss + anchor_loss
    return LossBreakdown(total=total, e_loss=e_loss, q_loss=q_loss, residual_loss=residual_loss, anchor_loss=anchor_loss)
