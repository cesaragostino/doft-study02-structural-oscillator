"""Simulation primitives for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import math

from .data import DELTA_KEYS, PRIMES, PRIME_KEYS, SubnetParameters


@dataclass
class SimulationResult:
    """Container for a simulated fingerprint."""

    e_sim: List[float]
    q_sim: float | None
    residual_sim: float
    layer_factors: List[float]


def soft_round(value: float, softness: float = 0.35) -> float:
    """Return a smooth approximation of the nearest integer."""

    nearest = round(value)
    diff = value - nearest
    return nearest + diff * math.tanh(abs(diff) / max(softness, 1e-6))


def _layer_factor(layer_index: int) -> float:
    """Compute a deterministic factor for the given layer index."""

    return 1.0 + 0.18 * (layer_index - 1)


class ClusterSimulator:
    """Compute fingerprints for a set of simulation parameters."""

    def __init__(self, softness: float = 0.35) -> None:
        self.softness = softness

    def simulate(self, params: SubnetParameters) -> SimulationResult:
        e_sim: List[float] = []
        layer_factors: List[float] = []
        for idx, prime in enumerate(PRIMES):
            key = PRIME_KEYS[idx]
            delta_key = DELTA_KEYS[idx]
            ratio = params.ratios.get(key, 0.0)
            delta = params.delta.get(delta_key, 0.0)
            layer_index = params.layer_assignment[idx] if idx < len(params.layer_assignment) else 1
            layer_index = max(1, min(params.L, layer_index))
            layer_factor = _layer_factor(layer_index)
            base_value = params.f0 * (1.0 + ratio) * layer_factor + delta
            e_value = soft_round(base_value, self.softness)
            e_sim.append(e_value)
            layer_factors.append(layer_factor)

        q_sim = self._compute_q(e_sim)
        residual_sim = self._compute_residual(params.f0, e_sim, params.delta.values())
        return SimulationResult(e_sim=e_sim, q_sim=q_sim, residual_sim=residual_sim, layer_factors=layer_factors)

    def _compute_q(self, e_sim: Sequence[float]) -> float | None:
        weights = [max(e, 0.0) for e in e_sim]
        total = sum(weights)
        if total <= 0:
            return None
        numerator = sum(weight * prime for weight, prime in zip(weights, PRIMES))
        return numerator / total

    def _compute_residual(self, f0: float, e_sim: Iterable[float], deltas: Iterable[float]) -> float:
        e_list = list(e_sim)
        avg_e = sum(e_list) / max(len(e_list), 1)
        delta_list = list(deltas)
        avg_delta = sum(abs(delta) for delta in delta_list) / max(len(delta_list), 1) if delta_list else 0.0
        log_prime = sum(e * math.log(p) for e, p in zip(e_list, PRIMES)) / max(sum(e_list) + 1e-6, 1e-6)
        return math.log(max(f0, 1e-6)) - log_prime - 0.05 * avg_delta + 0.01 * avg_e

    # Expose optimizable parameters for downstream optimizers
    def get_optimizable_parameters(self, params: SubnetParameters) -> Dict[str, float]:
        values = {
            "f0": params.f0,
            "r2": params.ratios.get("r2", 0.0),
            "r3": params.ratios.get("r3", 0.0),
            "r5": params.ratios.get("r5", 0.0),
            "r7": params.ratios.get("r7", 0.0),
            "d2": params.delta.get("d2", 0.0),
            "d3": params.delta.get("d3", 0.0),
            "d5": params.delta.get("d5", 0.0),
            "d7": params.delta.get("d7", 0.0),
        }
        delta_T = getattr(params, "delta_T", {}) or {}
        delta_space = getattr(params, "delta_space", {}) or {}
        for prime in PRIMES:
            key = str(prime)
            values[f"delta_T_{key}"] = delta_T.get(key, 0.0)
            values[f"delta_space_{key}"] = delta_space.get(key, 0.0)
        return values

    def set_optimizable_parameters(self, params: SubnetParameters, values: Dict[str, float]) -> None:
        params.f0 = values.get("f0", params.f0)
        for key in ("r2", "r3", "r5", "r7"):
            if key in values:
                params.ratios[key] = values[key]
        for key in ("d2", "d3", "d5", "d7"):
            if key in values:
                params.delta[key] = values[key]
        for prime in PRIMES:
            p_key = str(prime)
            key_T = f"delta_T_{p_key}"
            key_space = f"delta_space_{p_key}"
            if key_T in values:
                params.delta_T[p_key] = values[key_T]
            if key_space in values:
                params.delta_space[p_key] = values[key_space]
