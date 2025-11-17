<<<<<<< ours
<<<<<<< ours
"""Adaptive optimizer for DOFT cluster simulation parameters."""
=======
"""Random + local search optimizer for DOFT cluster simulation."""
>>>>>>> theirs
=======
"""Random + local search optimizer for DOFT cluster simulation."""
>>>>>>> theirs

from __future__ import annotations

from dataclasses import dataclass
<<<<<<< ours
<<<<<<< ours
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random

import numpy as np

from .data import (
    DELTA_KEYS,
    ParameterBounds,
    PRIME_KEYS,
    PRIMES,
    LossWeights,
    SubnetConfig,
    SubnetParameters,
    SubnetTarget,
)
=======
=======
>>>>>>> theirs
from typing import Optional, Tuple
import random

from .data import DELTA_KEYS, PRIME_KEYS, LossWeights, SubnetParameters, SubnetTarget
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
from .loss import LossBreakdown, compute_subnet_loss
from .model import ClusterSimulator


<<<<<<< ours
<<<<<<< ours
RATIO_KEY_BY_PRIME = {prime: key for prime, key in zip(PRIMES, PRIME_KEYS)}
DELTA_KEY_BY_PRIME = {prime: key for prime, key in zip(PRIMES, DELTA_KEYS)}


=======
>>>>>>> theirs
=======
>>>>>>> theirs
@dataclass
class OptimizationResult:
    """Optimization outcome for a subnet."""

    params: SubnetParameters
    simulation_loss: LossBreakdown
    simulation_result: "SimulationResult"


class SubnetOptimizer:
<<<<<<< ours
<<<<<<< ours
    """Gradient-based search (finite-difference Adam) for subnet parameters."""
=======
    """Search for parameters that minimise the loss for a subnet."""
>>>>>>> theirs
=======
    """Search for parameters that minimise the loss for a subnet."""
>>>>>>> theirs

    def __init__(
        self,
        simulator: ClusterSimulator,
        weights: LossWeights,
<<<<<<< ours
<<<<<<< ours
        bounds: ParameterBounds,
        max_steps: int,
        rng: random.Random,
        anchor: Optional[float],
        subnet_config: Optional[SubnetConfig],
        freeze_primes: Iterable[int],
        active_primes: Sequence[int],
        lambda_reg: float,
        prime_layers: Sequence[int],
        seed: int,
        thermal_scale: float,
        eta: float,
        prime_value: Optional[float],
        ratio_bounds: Tuple[float, float],
        delta_bounds: Tuple[float, float],
        ratio_bounds_by_key: Dict[str, Tuple[float, float]],
        delta_bounds_by_key: Dict[str, Tuple[float, float]],
    ) -> None:
        self.simulator = simulator
        self.weights = weights
        self.bounds = bounds
        self.max_steps = max(1, max_steps)
        self.anchor = anchor
        self.subnet_config = subnet_config
        self.freeze_primes = tuple(sorted({int(p) for p in freeze_primes}))
        self.active_primes = tuple(sorted({int(p) for p in active_primes}))
        self.lambda_reg = lambda_reg
        self.prime_layers = list(prime_layers) if prime_layers else [1 for _ in PRIMES]
        self.layer = max(1, max(self.prime_layers))
        self.seed_rng = random.Random(seed)
        self.rng = rng
        self.thermal_scale = thermal_scale
        self.eta = eta
        self.prime_value = prime_value
        self.ratio_bounds = ratio_bounds
        self.delta_bounds = delta_bounds
        self.ratio_bounds_by_key = ratio_bounds_by_key
        self.delta_bounds_by_key = delta_bounds_by_key

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.learning_rate = 1e-2
        self.min_learning_rate = 1e-4
        self.lr_factor = 0.5
        self.lr_plateau_patience = 20
        self.early_stopping_patience = 50
        self.min_delta = 1e-5
        self.grad_clip = 1.0
        self.fd_epsilon = 1e-3

        self.active_ratio_keys = self._select_keys(RATIO_KEY_BY_PRIME)
        self.active_delta_keys = self._select_keys(DELTA_KEY_BY_PRIME)
        self.vector_size = 1 + len(self.active_ratio_keys) + len(self.active_delta_keys)

    def optimise(self, target: SubnetTarget, target_key: str) -> OptimizationResult:
        vector = self._initial_vector()
        m = np.zeros(self.vector_size)
        v = np.zeros(self.vector_size)
        lr = self.learning_rate

        best_params: Optional[SubnetParameters] = None
        best_loss: Optional[LossBreakdown] = None
        best_result = None
        best_vector = vector.copy()
        no_improve_steps = 0
        plateau_steps = 0

        for step in range(1, self.max_steps + 1):
            loss, simulation_result, params = self._evaluate_vector(vector, target, target_key)
            improved = best_loss is None or (loss.total + self.min_delta) < best_loss.total
            if improved:
                best_loss = loss
                best_params = params
                best_result = simulation_result
                best_vector = vector.copy()
                no_improve_steps = 0
                plateau_steps = 0
            else:
                no_improve_steps += 1
                plateau_steps += 1

            if no_improve_steps >= self.early_stopping_patience and lr <= self.min_learning_rate + 1e-9:
                break

            if plateau_steps >= self.lr_plateau_patience:
                lr = max(self.min_learning_rate, lr * self.lr_factor)
                plateau_steps = 0

            grad = self._finite_difference_gradient(vector, target, target_key)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > self.grad_clip and grad_norm > 0:
                grad = grad * (self.grad_clip / grad_norm)

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            m_hat = m / (1.0 - self.beta1 ** step)
            v_hat = v / (1.0 - self.beta2 ** step)

            vector = vector - lr * (m_hat / (np.sqrt(v_hat) + 1e-8))
            vector = self._clamp_vector(vector)

        if best_params is None or best_loss is None or best_result is None:
            # Fallback to last evaluation
            best_loss, best_result, best_params = self._evaluate_vector(best_vector, target, target_key)

        return OptimizationResult(params=best_params, simulation_loss=best_loss, simulation_result=best_result)

    # ---- internal helpers -------------------------------------------------

    def _select_keys(self, mapping: dict[int, str]) -> List[str]:
        selected: List[str] = []
        freeze_set = set(self.freeze_primes)
        for prime in self.active_primes:
            if prime in freeze_set:
                continue
            key = mapping.get(prime)
            if key:
                selected.append(key)
        return selected

    def _initial_vector(self) -> np.ndarray:
        vector = np.zeros(self.vector_size, dtype=float)
        anchor = self.anchor if self.anchor is not None else self._default_f0()
        vector[0] = self._clamp(anchor, self.bounds.f0)
        noise_scale = 0.01
        idx = 1
        for _ in self.active_ratio_keys:
            vector[idx] = self.seed_rng.gauss(0.0, noise_scale)
            idx += 1
        for _ in self.active_delta_keys:
            vector[idx] = self.seed_rng.gauss(0.0, noise_scale)
            idx += 1
        return self._clamp_vector(vector)

    def _default_f0(self) -> float:
        lo, hi = self.bounds.f0
        return 0.5 * (lo + hi)

    def _clamp(self, value: float, bounds: Tuple[float, float]) -> float:
        lo, hi = bounds
        return min(max(value, lo), hi)

    def _clamp_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = vector.copy()
        vector[0] = self._clamp(float(vector[0]), self.bounds.f0)
        idx = 1
        for key in self.active_ratio_keys:
            bounds = self.ratio_bounds_by_key.get(key, self.ratio_bounds)
            vector[idx] = self._clamp(float(vector[idx]), bounds)
            idx += 1
        for key in self.active_delta_keys:
            bounds = self.delta_bounds_by_key.get(key, self.delta_bounds)
            vector[idx] = self._clamp(float(vector[idx]), bounds)
            idx += 1
        return vector

    def _vector_to_params(self, vector: np.ndarray) -> SubnetParameters:
        ratios = {key: 0.0 for key in PRIME_KEYS}
        deltas = {key: 0.0 for key in DELTA_KEYS}
        idx = 1
        for key in self.active_ratio_keys:
            ratios[key] = float(vector[idx])
            idx += 1
        for key in self.active_delta_keys:
            deltas[key] = float(vector[idx])
            idx += 1
        layer_assignment = list(self.prime_layers)
        return SubnetParameters(L=self.layer, f0=float(vector[0]), ratios=ratios, delta=deltas, layer_assignment=layer_assignment)

    def _evaluate_vector(
        self, vector: np.ndarray, target: SubnetTarget, target_key: str
    ) -> Tuple[LossBreakdown, "SimulationResult", SubnetParameters]:
        params = self._vector_to_params(vector)
        simulation_result = self.simulator.simulate(params)
        loss = compute_subnet_loss(
            target=target,
            params=params,
            simulation=simulation_result,
            weights=self.weights,
            anchor_value=self.anchor,
            subnet_name=target_key,
            thermal_scale=self.thermal_scale,
            eta=self.eta,
            prime_value=self.prime_value,
            lambda_reg=self.lambda_reg,
            active_ratio_keys=self.active_ratio_keys,
            active_delta_keys=self.active_delta_keys,
        )
        return loss, simulation_result, params

    def _finite_difference_gradient(self, vector: np.ndarray, target: SubnetTarget, target_key: str) -> np.ndarray:
        grad = np.zeros_like(vector)
        for idx in range(len(vector)):
            eps = self.fd_epsilon
            vector[idx] += eps
            loss_plus = self._evaluate_vector(vector, target, target_key)[0].total
            vector[idx] -= 2 * eps
            loss_minus = self._evaluate_vector(vector, target, target_key)[0].total
            vector[idx] += eps
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        return grad
=======
=======
>>>>>>> theirs
        max_evals: int,
        rng: random.Random,
        anchor: Optional[float],
    ) -> None:
        self.simulator = simulator
        self.weights = weights
        self.max_evals = max_evals
        self.rng = rng
        self.anchor = anchor

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
        loss = compute_subnet_loss(target, params, simulation_result, self.weights, self.anchor)
        return loss, simulation_result

    def _sample_parameters(self) -> SubnetParameters:
        L = self.rng.randint(1, 3)
        anchor = self.anchor if self.anchor is not None else 1.5
        f0 = max(0.25, self.rng.gauss(anchor, 0.4))
        ratios = {key: max(0.0, self.rng.uniform(0.0, 1.5)) for key in PRIME_KEYS}
        delta = {key: self.rng.uniform(-0.2, 0.2) for key in DELTA_KEYS}
        layer_assignment = [self.rng.randint(1, L) for _ in PRIME_KEYS]
        return SubnetParameters(L=L, f0=f0, ratios=ratios, delta=delta, layer_assignment=layer_assignment)

    def _perturb(self, params: SubnetParameters) -> SubnetParameters:
        candidate = params.copy()
        candidate.L = max(1, min(3, candidate.L + self.rng.choice([-1, 0, 1])))
        candidate.f0 = max(0.2, candidate.f0 + self.rng.gauss(0.0, 0.05))
        for key in PRIME_KEYS:
            candidate.ratios[key] = max(0.0, candidate.ratios[key] + self.rng.gauss(0.0, 0.05))
        for key in DELTA_KEYS:
            candidate.delta[key] = candidate.delta[key] + self.rng.gauss(0.0, 0.02)
        candidate.layer_assignment = [
            max(1, min(candidate.L, layer + self.rng.choice([-1, 0, 1]))) for layer in candidate.layer_assignment
        ]
        return candidate

<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
