"""Random + local search optimizer for DOFT cluster simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import random

from .data import DELTA_KEYS, PRIME_KEYS, LossWeights, ParameterBounds, SubnetParameters, SubnetTarget
from .loss import LossBreakdown, compute_subnet_loss
from .model import ClusterSimulator


@dataclass
class OptimizationResult:
    """Optimization outcome for a subnet."""

    params: SubnetParameters
    simulation_loss: LossBreakdown
    simulation_result: "SimulationResult"


class SubnetOptimizer:
    """Search for parameters that minimise the loss for a subnet."""

    def __init__(
        self,
        simulator: ClusterSimulator,
        weights: LossWeights,
        bounds: ParameterBounds,
        max_evals: int,
        rng: random.Random,
        anchor: Optional[float],
        huber_delta: Optional[float],
        xi_value: Optional[float],
        xi_sign: int,
        xi_exp: Optional[dict],
        k_skin: float,
        base_delta_T: float = 0.0,
        base_delta_space: float = 0.0,
    ) -> None:
        self.simulator = simulator
        self.weights = weights
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = rng
        self.anchor = anchor
        self.huber_delta = huber_delta
        self.xi_value = xi_value
        self.xi_sign = xi_sign
        self.xi_exp = xi_exp or {}
        self.k_skin = k_skin
        self.base_delta_T = base_delta_T
        self.base_delta_space = base_delta_space

    def optimise(self, target: SubnetTarget) -> OptimizationResult:
        best_params: Optional[SubnetParameters] = None
        best_loss: Optional[LossBreakdown] = None
        best_result = None

        for _ in range(self.max_evals):
            candidate = self._sample_parameters()
            loss, simulation_result = self._evaluate(target, candidate)
            if best_loss is None or loss.total < best_loss.total:
                best_params = candidate
                best_loss = loss
                best_result = simulation_result

        # Local refinement around the best candidate
        if best_params is None or best_loss is None or best_result is None:
            raise RuntimeError("No candidate evaluated during optimisation")

        for _ in range(min(64, self.max_evals // 2 + 1)):
            candidate = self._perturb(best_params)
            loss, simulation_result = self._evaluate(target, candidate)
            if loss.total < best_loss.total:
                best_params = candidate
                best_loss = loss
                best_result = simulation_result

        return OptimizationResult(params=best_params, simulation_loss=best_loss, simulation_result=best_result)

    def _evaluate(self, target: SubnetTarget, params: SubnetParameters) -> Tuple[LossBreakdown, "SimulationResult"]:
        simulation_result = self.simulator.simulate(params)
        loss = compute_subnet_loss(
            target=target,
            params=params,
            simulation=simulation_result,
            weights=self.weights,
            anchor_value=self.anchor,
            huber_delta=self.huber_delta,
            xi_value=self.xi_value,
            xi_sign=self.xi_sign,
            xi_exp=self.xi_exp,
            k_skin=self.k_skin,
            delta_T=params.delta_T,
            delta_space=params.delta_space,
        )
        return loss, simulation_result

    def _sample_parameters(self) -> SubnetParameters:
        L = self.rng.randint(1, 3)
        anchor = self.anchor if self.anchor is not None else 0.5 * sum(self.bounds.f0)
        f0 = self._clamp(self.rng.gauss(anchor, 0.4), self.bounds.f0)
        ratios = {key: self.rng.uniform(*self.bounds.ratios) for key in PRIME_KEYS}
        delta = {key: self.rng.uniform(*self.bounds.deltas) for key in DELTA_KEYS}
        layer_assignment = [self.rng.randint(1, L) for _ in PRIME_KEYS]
        delta_T = self.base_delta_T + self.rng.gauss(0.0, 0.01)
        delta_space = self.base_delta_space + self.rng.gauss(0.0, 0.01)
        return SubnetParameters(
            L=L, f0=f0, ratios=ratios, delta=delta, layer_assignment=layer_assignment, delta_T=delta_T, delta_space=delta_space
        )

    def _perturb(self, params: SubnetParameters) -> SubnetParameters:
        candidate = params.copy()
        candidate.L = max(1, min(3, candidate.L + self.rng.choice([-1, 0, 1])))
        candidate.f0 = self._clamp(candidate.f0 + self.rng.gauss(0.0, 0.05), self.bounds.f0)
        for key in PRIME_KEYS:
            candidate.ratios[key] = self._clamp(candidate.ratios[key] + self.rng.gauss(0.0, 0.05), self.bounds.ratios)
        for key in DELTA_KEYS:
            candidate.delta[key] = self._clamp(candidate.delta[key] + self.rng.gauss(0.0, 0.02), self.bounds.deltas)
        candidate.delta_T = candidate.delta_T + self.rng.gauss(0.0, 0.01)
        candidate.delta_space = candidate.delta_space + self.rng.gauss(0.0, 0.01)
        candidate.layer_assignment = [
            max(1, min(candidate.L, layer + self.rng.choice([-1, 0, 1]))) for layer in candidate.layer_assignment
        ]
        return candidate

    @staticmethod
    def _clamp(value: float, bounds: Tuple[float, float]) -> float:
        lo, hi = bounds
        return min(max(value, lo), hi)
