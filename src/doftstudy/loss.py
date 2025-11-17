"""Loss computation utilities for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass
<<<<<<< ours
<<<<<<< ours
from typing import Dict, Iterable, List, Optional
import math

from .data import DELTA_KEYS, LossWeights, PRIME_KEYS, SubnetParameters, SubnetTarget
=======
from typing import Dict, List, Optional

from .data import LossWeights, PRIMES, SubnetParameters, SubnetTarget
>>>>>>> theirs
=======
from typing import Dict, List, Optional

from .data import LossWeights, PRIMES, SubnetParameters, SubnetTarget
>>>>>>> theirs
from .model import SimulationResult


@dataclass
class LossBreakdown:
    """Detailed loss information for a subnet."""

    total: float
    e_loss: float
    q_loss: float
    residual_loss: float
    anchor_loss: float
<<<<<<< ours
<<<<<<< ours
    regularization_loss: float
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "e": self.e_loss,
            "q": self.q_loss,
            "residual": self.residual_loss,
            "anchor": self.anchor_loss,
<<<<<<< ours
<<<<<<< ours
            "regularization": self.regularization_loss,
        }


=======
=======
>>>>>>> theirs
        }


def _safe_len(values: Optional[List[Optional[float]]]) -> int:
    if values is None:
        return 0
    return sum(1 for value in values if value is not None)


<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
def compute_subnet_loss(
    target: SubnetTarget,
    params: SubnetParameters,
    simulation: SimulationResult,
    weights: LossWeights,
    anchor_value: Optional[float],
<<<<<<< ours
<<<<<<< ours
    subnet_name: str,
    thermal_scale: float = 0.0,
    eta: float = 0.0,
    prime_value: Optional[float] = None,
    lambda_reg: float = 0.0,
    active_ratio_keys: Optional[Iterable[str]] = None,
    active_delta_keys: Optional[Iterable[str]] = None,
) -> LossBreakdown:
    """Compute the weighted loss for a subnet."""

    ratio_keys = list(active_ratio_keys) if active_ratio_keys is not None else list(PRIME_KEYS)
    delta_keys = list(active_delta_keys) if active_delta_keys is not None else list(DELTA_KEYS)

=======
) -> LossBreakdown:
    """Compute the weighted loss for a subnet."""

>>>>>>> theirs
=======
) -> LossBreakdown:
    """Compute the weighted loss for a subnet."""

>>>>>>> theirs
    e_terms = []
    if target.e_exp is not None:
        for idx, value in enumerate(target.e_exp):
            if value is None:
                continue
<<<<<<< ours
<<<<<<< ours
            diff = abs(simulation.e_sim[idx] - value)
            e_terms.append(diff)
    e_loss = (sum(e_terms) / len(e_terms)) * weights.w_e if e_terms else 0.0

    q_loss = 0.0
    q_weight = weights.q_weight_for(subnet_name, target.use_q)
    if q_weight > 0.0 and target.q_exp is not None and simulation.q_sim is not None:
        diff_q = abs(simulation.q_sim - target.q_exp)
        q_loss = diff_q * q_weight

    residual_loss = 0.0
    residual_value = simulation.residual_sim
    if prime_value is not None:
        residual_value = _compute_residual(simulation.log_r, thermal_scale, eta, prime_value)
        simulation.residual_sim = residual_value
    if target.residual_exp is not None and prime_value is not None:
        diff_r = abs(residual_value - target.residual_exp)
        residual_loss = diff_r * weights.w_r
=======
=======
>>>>>>> theirs
            diff = simulation.e_sim[idx] - value
            e_terms.append(diff * diff)
    e_loss = (sum(e_terms) / max(len(e_terms), 1)) * weights.w_e if e_terms else 0.0

    q_loss = 0.0
    if target.q_exp is not None and simulation.q_sim is not None:
        diff_q = simulation.q_sim - target.q_exp
        q_loss = (diff_q * diff_q) * weights.w_q

    residual_loss = 0.0
    if target.residual_exp is not None:
        diff_r = simulation.residual_sim - target.residual_exp
        residual_loss = (diff_r * diff_r) * weights.w_r
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs

    anchor_loss = 0.0
    if anchor_value is not None:
        diff_a = params.f0 - anchor_value
        anchor_loss = (diff_a * diff_a) * weights.w_anchor

<<<<<<< ours
<<<<<<< ours
    reg_terms = [params.ratios.get(key, 0.0) ** 2 for key in ratio_keys]
    reg_terms += [params.delta.get(key, 0.0) ** 2 for key in delta_keys]
    regularization_loss = lambda_reg * sum(reg_terms)

    total = e_loss + q_loss + residual_loss + anchor_loss + regularization_loss
    return LossBreakdown(
        total=total,
        e_loss=e_loss,
        q_loss=q_loss,
        residual_loss=residual_loss,
        anchor_loss=anchor_loss,
        regularization_loss=regularization_loss,
    )


def _compute_residual(log_r: float, thermal_scale: float, eta: float, prime_value: float) -> float:
    eps = 1e-12
    log_corr = log_r - eta * thermal_scale
    return log_corr - math.log(max(prime_value, eps))
=======
    total = e_loss + q_loss + residual_loss + anchor_loss
    return LossBreakdown(total=total, e_loss=e_loss, q_loss=q_loss, residual_loss=residual_loss, anchor_loss=anchor_loss)

>>>>>>> theirs
=======
    total = e_loss + q_loss + residual_loss + anchor_loss
    return LossBreakdown(total=total, e_loss=e_loss, q_loss=q_loss, residual_loss=residual_loss, anchor_loss=anchor_loss)

>>>>>>> theirs
