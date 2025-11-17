"""High level orchestration for the DOFT cluster simulator."""

from __future__ import annotations

from typing import Dict, Optional
import random

from .data import LossWeights, MaterialConfig, ParameterBounds, TargetDataset
from .model import ClusterSimulator
from .optimizer import SubnetOptimizer
from .reporting import ReportBundle, create_report_bundle
from .results import SubnetSimulation


class SimulationEngine:
    """Orchestrate optimisation across subnets and produce reports."""

    def __init__(
        self,
        config: MaterialConfig,
        dataset: TargetDataset,
        weights: LossWeights,
        max_evals: int = 250,
        seed: int = 42,
        bounds: Optional[ParameterBounds] = None,
        huber_delta: Optional[float] = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.weights = weights
        self.max_evals = max_evals
        self.rng = random.Random(seed)
        self.simulator = ClusterSimulator()
        self.bounds = bounds or ParameterBounds()
        self.huber_delta = huber_delta

    def run(self) -> ReportBundle:
        subnet_results: Dict[str, SubnetSimulation] = {}
        for subnet_name in self.config.subnets:
            target = self.dataset.subnets.get(f"{self.config.material}_{subnet_name}")
            if target is None:
                raise KeyError(f"Missing target for subnet '{subnet_name}'")
            anchor = self._lookup_anchor(subnet_name)
            optimizer = SubnetOptimizer(
                simulator=self.simulator,
                weights=self.weights,
                bounds=self.bounds,
                max_evals=self.max_evals,
                rng=self.rng,
                anchor=anchor,
                huber_delta=self.huber_delta,
            )
            result = optimizer.optimise(target)
            subnet_results[subnet_name] = SubnetSimulation(
                parameters=result.params,
                loss=result.simulation_loss,
                simulation_result=result.simulation_result,
            )

        bundle = create_report_bundle(
            config=self.config,
            dataset=self.dataset,
            weights=self.weights,
            subnet_results=subnet_results,
        )
        return bundle

    def _lookup_anchor(self, subnet_name: str) -> Optional[float]:
        anchor_data = self.config.anchors.get(subnet_name)
        if anchor_data is None:
            return None
        return anchor_data.get("X")
