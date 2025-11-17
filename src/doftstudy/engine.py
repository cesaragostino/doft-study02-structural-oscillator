"""High level orchestration for the DOFT cluster simulator."""

from __future__ import annotations

<<<<<<< ours
<<<<<<< ours
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random
import uuid

from .data import ContrastTarget, LossWeights, MaterialConfig, ParameterBounds, TargetDataset
from .model import ClusterSimulator
from .optimizer import SubnetOptimizer
from .reporting import ReportBundle, create_report_bundle
from .results import SimulationRun, SubnetSimulation

SCHEMA_VERSION = "0.2"
=======
=======
>>>>>>> theirs
from typing import Dict, Optional
import random

from .data import LossWeights, MaterialConfig, TargetDataset
from .model import ClusterSimulator
from .optimizer import SubnetOptimizer
from .reporting import ReportBundle, create_report_bundle
from .results import SubnetSimulation
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs


class SimulationEngine:
    """Orchestrate optimisation across subnets and produce reports."""

    def __init__(
        self,
        config: MaterialConfig,
        dataset: TargetDataset,
        weights: LossWeights,
        max_evals: int = 250,
        seed: int = 42,
<<<<<<< ours
<<<<<<< ours
        freeze_primes: Optional[Iterable[int]] = None,
        ablation_sets: Optional[List[Sequence[int]]] = None,
        seed_sweep: int = 1,
        bounds_override: Optional[Dict[str, Tuple[float, float]]] = None,
        huber_delta: float = 0.02,
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.weights = weights
        self.max_evals = max_evals
<<<<<<< ours
<<<<<<< ours
        self.base_seed = seed
        self.seed_sweep = max(1, seed_sweep)
        self.huber_delta = huber_delta
        self.simulator = ClusterSimulator()

        self.rng = random.Random(seed)
        self.bounds = self._resolve_bounds(bounds_override)
        self.global_freeze_primes = set(config.freeze_primes)
        if freeze_primes is not None:
            self.global_freeze_primes.update(int(p) for p in freeze_primes)
        self.prime_layers = config.prime_layers
        self.eta = config.eta

        if ablation_sets:
            self.ablation_sets = [tuple(int(p) for p in subset) for subset in ablation_sets if subset]
        else:
            self.ablation_sets = [tuple(config.primes)]
        if not self.ablation_sets:
            self.ablation_sets = [tuple(config.primes)]

    def run(self) -> ReportBundle:
        runs: List[SimulationRun] = []
        contrast_targets = self._collect_contrasts()
        run_counter = 0
        seed_history: List[int] = []
        run_identifier = uuid.uuid4().hex

        for prime_subset in self.ablation_sets:
            active_primes = tuple(int(p) for p in prime_subset if p in self.config.primes)
            if not active_primes:
                active_primes = tuple(self.config.primes)
            freeze_set = self._derive_freeze_set(active_primes)
            for sweep_idx in range(self.seed_sweep):
                run_seed = self.base_seed + run_counter * 997 + sweep_idx
                seed_history.append(run_seed)
                run_label = self._build_run_label(run_seed, active_primes)
                subnet_results, base_loss = self._optimise_subnets(
                    active_primes=active_primes,
                    freeze_primes=freeze_set,
                    run_seed=run_seed,
                )
                runs.append(
                    SimulationRun(
                        label=run_label,
                        seed=run_seed,
                        primes=active_primes,
                        freeze_primes=tuple(sorted(freeze_set)),
                        subnet_results=subnet_results,
                        base_loss=base_loss,
                    )
                )
                run_counter += 1

        bundle = create_report_bundle(
            config=self.config,
            weights=self.weights,
            runs=runs,
            contrast_targets=contrast_targets,
            dataset=self.dataset,
            seed_list=seed_history,
            seed_sweep=self.seed_sweep,
            run_id=run_identifier,
            schema_version=SCHEMA_VERSION,
            bounds=self.bounds,
        )
        return bundle

    def _lookup_anchor(self, subnet_name: str) -> Optional[float]:
        subnet_cfg = self.config.subnet_configs.get(subnet_name)
        if subnet_cfg and subnet_cfg.f0_anchor is not None:
            return subnet_cfg.f0_anchor
        anchor_data = self.config.anchors.get(subnet_name)
        if anchor_data is None:
            return None
        return anchor_data.get("f0") or anchor_data.get("X")

    def _collect_contrasts(self) -> List[ContrastTarget]:
        merged: Dict[tuple[str, str], ContrastTarget] = {}
        for contrast in self.config.contrasts:
            merged[(contrast.subnet_a, contrast.subnet_b)] = contrast
        for contrast in self.dataset.contrasts:
            key = (contrast.subnet_a, contrast.subnet_b)
            if contrast.label is None and key in merged and merged[key].label is not None:
                contrast.label = merged[key].label
            merged[key] = contrast
        return list(merged.values())

    def _derive_freeze_set(self, active_primes: Sequence[int]) -> set[int]:
        freeze_set = set(self.global_freeze_primes)
        config_primes = set(self.config.primes)
        active = set(active_primes)
        freeze_set.update(config_primes - active)
        return freeze_set

    def _optimise_subnets(
        self,
        active_primes: Sequence[int],
        freeze_primes: Iterable[int],
        run_seed: int,
    ) -> Tuple[Dict[str, SubnetSimulation], float]:
        subnet_results: Dict[str, SubnetSimulation] = {}
        base_loss = 0.0
        for idx, subnet_name in enumerate(self.config.subnets):
            subnet_cfg = self.config.subnet_configs.get(subnet_name)
            if subnet_cfg is not None and not subnet_cfg.enabled:
                continue
            target_key = f"{self.config.material}_{subnet_name}"
            target = self.dataset.subnets.get(target_key)
            if target is None:
                raise KeyError(f"Missing target for subnet '{subnet_name}'")
            anchor = self._lookup_anchor(subnet_name)
            prime_layers = subnet_cfg.prime_layers if subnet_cfg else self.prime_layers
            thermal_scale = subnet_cfg.thermal_scale if subnet_cfg else self.config.thermal_scales.get(subnet_name, 0.0)
            ratio_bounds, delta_bounds, ratio_map, delta_map = self._build_bound_maps(subnet_cfg)
            run_rng = random.Random(run_seed + idx * 17)
            optimizer = SubnetOptimizer(
                simulator=self.simulator,
                weights=self.weights,
                bounds=self.bounds,
                max_steps=self.max_evals,
                rng=run_rng,
                anchor=anchor,
                subnet_config=subnet_cfg,
                freeze_primes=freeze_primes,
                active_primes=active_primes,
                lambda_reg=self.weights.lambda_reg,
                prime_layers=prime_layers,
                seed=run_seed + idx * 997,
                thermal_scale=thermal_scale,
                eta=self.eta,
                prime_value=target.prime_value,
                ratio_bounds=ratio_bounds,
                delta_bounds=delta_bounds,
                ratio_bounds_by_key=ratio_map,
                delta_bounds_by_key=delta_map,
            )
            result = optimizer.optimise(target, target_key)
=======
=======
>>>>>>> theirs
        self.rng = random.Random(seed)
        self.simulator = ClusterSimulator()

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
                max_evals=self.max_evals,
                rng=self.rng,
                anchor=anchor,
            )
            result = optimizer.optimise(target)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
            subnet_results[subnet_name] = SubnetSimulation(
                parameters=result.params,
                loss=result.simulation_loss,
                simulation_result=result.simulation_result,
            )
<<<<<<< ours
<<<<<<< ours
            base_loss += result.simulation_loss.total
        return subnet_results, base_loss

    def _build_run_label(self, seed: int, primes: Sequence[int]) -> str:
        primes_label = ",".join(str(p) for p in primes)
        return f"seed={seed}|primes={primes_label}"

    def _build_bound_maps(self, subnet_cfg: Optional['SubnetConfig']) -> Tuple[Tuple[float, float], Tuple[float, float], Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        ratio_bounds = self.bounds.ratios
        delta_bounds = self.bounds.deltas
        ratio_map: Dict[str, Tuple[float, float]] = {key: ratio_bounds for key in ('r2','r3','r5','r7')}
        delta_map: Dict[str, Tuple[float, float]] = {key: delta_bounds for key in ('d2','d3','d5','d7')}
        if subnet_cfg is not None:
            if subnet_cfg.ratio_bounds is not None:
                ratio_bounds = subnet_cfg.ratio_bounds
            if subnet_cfg.delta_bounds is not None:
                delta_bounds = subnet_cfg.delta_bounds
            for key, bounds in subnet_cfg.ratio_bounds_by_prime.items():
                ratio_map[key] = bounds
            for key, bounds in subnet_cfg.delta_bounds_by_prime.items():
                delta_map[key] = bounds
        else:
            ratio_map = {key: ratio_bounds for key in ('r2','r3','r5','r7')}
            delta_map = {key: delta_bounds for key in ('d2','d3','d5','d7')}
        return ratio_bounds, delta_bounds, ratio_map, delta_map

    def _resolve_bounds(self, override: Optional[Dict[str, Tuple[float, float]]]) -> ParameterBounds:
        if not override:
            return self.config.constraints
        ratios = override.get("ratios_bounds", self.config.constraints.ratios)
        deltas = override.get("deltas_bounds", self.config.constraints.deltas)
        f0 = override.get("f0_bounds", self.config.constraints.f0)
        return ParameterBounds.from_dict(
            {
                "ratios_bounds": ratios,
                "deltas_bounds": deltas,
                "f0_bounds": f0,
            }
        )
=======
=======
>>>>>>> theirs

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

<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
