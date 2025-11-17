"""Data structures and parsers for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
<<<<<<< ours
<<<<<<< ours
from typing import Any, Dict, Iterable, List, Optional, Tuple
=======
from typing import Dict, Iterable, List, Optional, Tuple
>>>>>>> theirs
=======
from typing import Dict, Iterable, List, Optional, Tuple
>>>>>>> theirs
import json
import math


PRIMES: Tuple[int, int, int, int] = (2, 3, 5, 7)
PRIME_KEYS: Tuple[str, str, str, str] = ("r2", "r3", "r5", "r7")
DELTA_KEYS: Tuple[str, str, str, str] = ("d2", "d3", "d5", "d7")
<<<<<<< ours
<<<<<<< ours
DEFAULT_PRIMES: Tuple[int, ...] = PRIMES
=======
>>>>>>> theirs
=======
>>>>>>> theirs


@dataclass
class LossWeights:
    """Collection of weights for each component of the loss function."""

    w_e: float = 1.0
    w_q: float = 0.5
    w_r: float = 0.25
    w_c: float = 0.3
    w_anchor: float = 0.05
<<<<<<< ours
<<<<<<< ours
    lambda_reg: float = 0.0
    overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, float]) -> "LossWeights":
        kwargs: Dict[str, object] = {}
        for key, value in data.items():
            if key == "overrides" and isinstance(value, dict):
                overrides: Dict[str, Dict[str, float]] = {}
                for override_key, override_value in value.items():
                    if not isinstance(override_value, dict):
                        continue
                    overrides[override_key] = {str(name): float(weight) for name, weight in override_value.items()}
                kwargs["overrides"] = overrides
            elif hasattr(cls, key):
                kwargs[key] = float(value)
        return cls(**kwargs)  # type: ignore[arg-type]

    def q_weight_for(self, subnet: str, use_q: bool) -> float:
        if not use_q:
            return 0.0
        override_block = self.overrides.get("q")
        if override_block and subnet in override_block:
            return float(override_block[subnet])
        return self.w_q


@dataclass
class ParameterBounds:
    """Global bounds applied to simulation parameters."""

    ratios: Tuple[float, float] = (-0.6, 0.6)
    deltas: Tuple[float, float] = (-0.8, 0.2)
    f0: Tuple[float, float] = (0.25, 50.0)

    @staticmethod
    def _clean_pair(values: Optional[Iterable[float]], default: Tuple[float, float]) -> Tuple[float, float]:
        if values is None:
            return default
        items = list(values)
        if len(items) != 2:
            return default
        lo = float(items[0])
        hi = float(items[1])
        if hi < lo:
            lo, hi = hi, lo
        if lo == hi:
            hi = lo + 1e-6
        return (lo, hi)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Iterable[float]]]) -> "ParameterBounds":
        if not isinstance(data, dict):
            return cls()
        ratios = cls._clean_pair(data.get("ratios_bounds"), cls.ratios)
        deltas = cls._clean_pair(data.get("deltas_bounds"), cls.deltas)
        f0 = cls._clean_pair(data.get("f0_bounds"), cls.f0)
        return cls(ratios=ratios, deltas=deltas, f0=f0)


@dataclass
class SubnetConfig:
    """Configuration hints and bounds for a subnet."""

    name: str
    enabled: bool = True
    l_candidates: List[int] = field(default_factory=lambda: [1])
    layer: int = 1
    prime_layers: List[int] = field(default_factory=lambda: [1 for _ in PRIMES])
    f0_anchor: Optional[float] = None
    f0_range: Optional[Tuple[float, float]] = None
    init_L: Optional[int] = None
    init_ratios: Dict[str, float] = field(default_factory=dict)
    ratio_abs_max: Optional[float] = None
    thermal_scale: float = 0.0
    ratio_bounds: Optional[Tuple[float, float]] = None
    delta_bounds: Optional[Tuple[float, float]] = None
    ratio_bounds_by_prime: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    delta_bounds_by_prime: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class OptimizationSettings:
    """Optional optimization hints sourced from the config file."""

    n_random_starts: Optional[int] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "OptimizationSettings":
        if not isinstance(data, dict):
            return cls()
        n_random = data.get("n_random_starts")
        seed = data.get("seed")
        return cls(
            n_random_starts=int(n_random) if isinstance(n_random, (int, float)) else None,
            seed=int(seed) if isinstance(seed, (int, float)) else None,
        )
=======
=======
>>>>>>> theirs

    @classmethod
    def from_json(cls, data: Dict[str, float]) -> "LossWeights":
        kwargs = {k: float(v) for k, v in data.items() if hasattr(cls, k)}
        return cls(**kwargs)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs


@dataclass
class SubnetTarget:
    """Target observables for a single subnet."""

    e_exp: Optional[List[Optional[float]]] = None
    q_exp: Optional[float] = None
    residual_exp: Optional[float] = None
    input_exponents: Optional[List[int]] = None
<<<<<<< ours
<<<<<<< ours
    prime_value: Optional[float] = None
    use_q: bool = True
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SubnetTarget":
        def _clean_e(values: Optional[Iterable[Optional[float]]]) -> Optional[List[Optional[float]]]:
            if values is None:
                return None
            cleaned: List[Optional[float]] = []
            for value in values:
                if value is None:
                    cleaned.append(None)
                    continue
                if isinstance(value, (int, float)) and math.isnan(value):
                    cleaned.append(None)
                else:
                    cleaned.append(float(value))
            return cleaned

        q_value = data.get("q_exp")
        if isinstance(q_value, (int, float)) and math.isnan(q_value):
            q_value = None

        residual_value = data.get("residual_exp")
        if isinstance(residual_value, (int, float)) and math.isnan(residual_value):
            residual_value = None

        input_exponents = data.get("input_exponents")
<<<<<<< ours
<<<<<<< ours
        prime_value = None
        if input_exponents is not None:
            input_exponents = [int(v) for v in input_exponents]
            prime_value = 1.0
            for exp, prime in zip(input_exponents, PRIMES):
                prime_value *= prime ** exp

        q_clean = None if q_value is None else float(q_value)
        if q_clean is not None:
            prime_value = q_clean
        return cls(
            e_exp=_clean_e(data.get("e_exp")),
            q_exp=q_clean,
            residual_exp=None if residual_value is None else float(residual_value),
            input_exponents=input_exponents,
            use_q=q_clean is not None,
            prime_value=prime_value,
=======
=======
>>>>>>> theirs
        if input_exponents is not None:
            input_exponents = [int(v) for v in input_exponents]

        return cls(
            e_exp=_clean_e(data.get("e_exp")),
            q_exp=None if q_value is None else float(q_value),
            residual_exp=None if residual_value is None else float(residual_value),
            input_exponents=input_exponents,
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
        )


@dataclass
class ContrastTarget:
    """Target contrast between two subnets."""

    subnet_a: str
    subnet_b: str
<<<<<<< ours
<<<<<<< ours
    value: Optional[float]
    label: Optional[str] = None
=======
    value: float
>>>>>>> theirs
=======
    value: float
>>>>>>> theirs


@dataclass
class SubnetParameters:
    """Simulation parameters for a subnet."""

    L: int
    f0: float
    ratios: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    layer_assignment: List[int] = field(default_factory=list)

    def copy(self) -> "SubnetParameters":
        return SubnetParameters(
            L=self.L,
            f0=self.f0,
            ratios=dict(self.ratios),
            delta=dict(self.delta),
            layer_assignment=list(self.layer_assignment),
        )


<<<<<<< ours
<<<<<<< ours
def _normalize_subnet_name(name: str, material: str) -> str:
    prefix = f"{material}_"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _normalize_ratio_key(key: str) -> Optional[str]:
    cleaned = str(key).strip().lower()
    if cleaned.startswith("r") and cleaned[1:].isdigit():
        cleaned = cleaned[1:]
    if cleaned in {"2", "3", "5", "7"}:
        return f"r{cleaned}"
    if cleaned in {"r2", "r3", "r5", "r7"}:
        return cleaned
    return None


def _normalize_ratio_mapping(values: Optional[Dict[str, object]]) -> Dict[str, float]:
    if not isinstance(values, dict):
        return {}
    normalized: Dict[str, float] = {}
    for key, value in values.items():
        prime_key = _normalize_ratio_key(str(key))
        if prime_key is None:
            continue
        normalized[prime_key] = float(value)
    return normalized


def _parse_prime_layers(raw_primes: object) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    prime_layer_map: Dict[int, int] = {prime: 1 for prime in PRIMES}
    selected_primes: List[int] = []

    if isinstance(raw_primes, dict):
        for prime in PRIMES:
            entry = raw_primes.get(str(prime)) or raw_primes.get(prime)
            if entry is None:
                continue
            selected_primes.append(prime)
            if isinstance(entry, dict) and "layer" in entry:
                layer_value = entry.get("layer")
                if isinstance(layer_value, (int, float)):
                    prime_layer_map[prime] = max(1, int(layer_value))
        if not selected_primes:
            selected_primes = list(PRIMES)
    elif isinstance(raw_primes, (list, tuple)):
        normalized = [int(value) for value in raw_primes if isinstance(value, (int, float))]
        selected_primes = [prime for prime in PRIMES if prime in normalized]
    else:
        selected_primes = list(PRIMES)

    if not selected_primes:
        selected_primes = list(PRIMES)

    return tuple(selected_primes), prime_layer_map


def _parse_subnet_configs(
    material: str,
    data: Dict[str, object],
    prime_layer_map: Dict[int, int],
) -> Tuple[Dict[str, SubnetConfig], List[str]]:
    subnet_configs: Dict[str, SubnetConfig] = {}
    enabled_order: List[str] = []

    raw_networks = data.get("sub_networks") or data.get("subnetworks")
    if isinstance(raw_networks, dict):
        items = raw_networks.items()
    elif isinstance(raw_networks, list):
        items = []
        for entry in raw_networks:
            if isinstance(entry, dict):
                name = entry.get("name")
                spec = entry
            else:
                name = entry
                spec = {}
            items.append((name, spec))
    else:
        items = []

    for raw_name, spec in items:
        if raw_name is None:
            continue
        name = _normalize_subnet_name(str(raw_name), material)
        subnet_configs[name] = _build_subnet_config(name, spec, prime_layer_map)
        if subnet_configs[name].enabled:
            enabled_order.append(name)

    raw_subnets = data.get("subnets") or data.get("subnetworks") or []
    if isinstance(raw_subnets, list):
        for entry in raw_subnets:
            name = _normalize_subnet_name(str(entry), material)
            if name not in subnet_configs:
                subnet_configs[name] = SubnetConfig(name=name, prime_layers=[prime_layer_map[p] for p in PRIMES])
            if subnet_configs[name].enabled and name not in enabled_order:
                enabled_order.append(name)

    if not subnet_configs and enabled_order:
        subnet_configs = {name: SubnetConfig(name=name, prime_layers=[prime_layer_map[p] for p in PRIMES]) for name in enabled_order}

    return subnet_configs, enabled_order


def _build_subnet_config(name: str, spec: object, prime_layer_map: Dict[int, int]) -> SubnetConfig:
    if not isinstance(spec, dict):
        spec = {}
    enabled = bool(spec.get("enabled", True))

    layer_value = spec.get("layer") or spec.get("L") or spec.get("layers")
    layer = max(1, int(layer_value)) if isinstance(layer_value, (int, float)) else 1

    raw_l = spec.get("L_candidates") or spec.get("l_candidates") or spec.get("L_options")
    l_candidates = _clean_l_candidates(raw_l)
    if not l_candidates:
        l_candidates = [1, 2, 3]

    f0_anchor = spec.get("f0_anchor", spec.get("X_anchor"))
    f0_anchor = float(f0_anchor) if isinstance(f0_anchor, (int, float)) else None

    f0_range = None
    raw_range = spec.get("f0_range") or spec.get("f0_bounds")
    if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
        lo = float(raw_range[0])
        hi = float(raw_range[1])
        if hi < lo:
            lo, hi = hi, lo
        if lo != hi:
            f0_range = (lo, hi)

    init_spec = spec.get("init")
    if not isinstance(init_spec, dict):
        init_spec = None
    init_L = None
    init_ratios: Dict[str, float] = {}
    if isinstance(init_spec, dict):
        init_L_value = init_spec.get("L") or init_spec.get("l")
        if isinstance(init_L_value, (int, float)):
            init_L = max(1, int(init_L_value))
        init_ratios = _normalize_ratio_mapping(init_spec.get("ratios"))

    bounds = spec.get("bounds") if isinstance(spec.get("bounds"), dict) else None
    ratio_abs_max = None
    ratio_bounds = None
    delta_bounds = None
    ratio_bounds_by_prime: Dict[str, Tuple[float, float]] = {}
    delta_bounds_by_prime: Dict[str, Tuple[float, float]] = {}
    if bounds:
        raw_ratio_max = bounds.get("ratio_abs_max")
        if isinstance(raw_ratio_max, (int, float)) and raw_ratio_max > 0:
            ratio_abs_max = float(raw_ratio_max)
        raw_ratio_bounds = bounds.get("ratio_bounds") or bounds.get("ratios_bounds")
        if isinstance(raw_ratio_bounds, (list, tuple)):
            ratio_bounds = ParameterBounds._clean_pair(raw_ratio_bounds, ParameterBounds.ratios)
        raw_delta_bounds = bounds.get("delta_bounds") or bounds.get("deltas_bounds")
        if isinstance(raw_delta_bounds, (list, tuple)):
            delta_bounds = ParameterBounds._clean_pair(raw_delta_bounds, ParameterBounds.deltas)
        raw_ratio_map = bounds.get("ratio_bounds_by_prime")
        if isinstance(raw_ratio_map, dict):
            for prime_key, value in raw_ratio_map.items():
                normalized = _normalize_ratio_key(f"r{prime_key}") or _normalize_ratio_key(str(prime_key))
                if normalized is None:
                    continue
                if isinstance(value, (list, tuple)):
                    ratio_bounds_by_prime[normalized] = ParameterBounds._clean_pair(value, ParameterBounds.ratios)
        raw_delta_map = bounds.get("delta_bounds_by_prime")
        if isinstance(raw_delta_map, dict):
            for prime_key, value in raw_delta_map.items():
                normalized = _normalize_ratio_key(f"d{prime_key}") or _normalize_ratio_key(f"d{prime_key}")
                if normalized is None:
                    continue
                if isinstance(value, (list, tuple)):
                    delta_bounds_by_prime[normalized] = ParameterBounds._clean_pair(value, ParameterBounds.deltas)

    prime_layers = [prime_layer_map[prime] for prime in PRIMES]
    layer = max(layer, max(prime_layers, default=1))

    return SubnetConfig(
        name=name,
        enabled=enabled,
        l_candidates=l_candidates or [layer],
        layer=layer,
        prime_layers=prime_layers,
        f0_anchor=f0_anchor,
        f0_range=f0_range,
        init_L=init_L,
        init_ratios=init_ratios,
        ratio_abs_max=ratio_abs_max,
        thermal_scale=0.0,
        ratio_bounds=ratio_bounds,
        delta_bounds=delta_bounds,
        ratio_bounds_by_prime=ratio_bounds_by_prime,
        delta_bounds_by_prime=delta_bounds_by_prime,
    )


def _clean_l_candidates(values: object) -> List[int]:
    if not isinstance(values, (list, tuple)):
        return []
    cleaned = []
    for value in values:
        if isinstance(value, (int, float)):
            cleaned.append(max(1, int(value)))
    deduped = sorted(set(cleaned))
    return deduped


def _parse_contrasts(material: str, subnets: List[str], raw_contrasts: object) -> List[ContrastTarget]:
    if raw_contrasts is None:
        return []
    results: List[ContrastTarget] = []
    subnet_set = set(subnets)

    def _append_contrast(a_name: str, b_name: str, value: Optional[object], label: Optional[str]) -> None:
        if value is None:
            return
        if a_name not in subnet_set or b_name not in subnet_set:
            raise ValueError(f"Contraste inválido: {a_name} o {b_name} no definidos en subnets")
        value_float = float(value)
        results.append(
            ContrastTarget(
                subnet_a=f"{material}_{a_name}",
                subnet_b=f"{material}_{b_name}",
                value=value_float,
                label=label,
            )
        )

    if isinstance(raw_contrasts, dict):
        for key, entry in raw_contrasts.items():
            if isinstance(entry, dict) and entry.get("enabled", True) is False:
                continue
            label = entry.get("type") if isinstance(entry, dict) else None
            value = None
            a_raw = None
            b_raw = None
            if isinstance(entry, dict):
                value = entry.get("target", entry.get("C_AB_exp"))
                a_raw = entry.get("A") or entry.get("a")
                b_raw = entry.get("B") or entry.get("b")
            if (a_raw is None or b_raw is None) and isinstance(key, str) and "_vs_" in key:
                parts = key.split("_vs_", 1)
                a_raw = parts[0]
                b_raw = parts[1]
            if a_raw is None or b_raw is None:
                raise ValueError("Cada contraste debe definir los campos 'A' y 'B' o usar una clave *_vs_*")
            a_name = _normalize_subnet_name(str(a_raw), material)
            b_name = _normalize_subnet_name(str(b_raw), material)
            label_value = str(label) if label else (key if isinstance(key, str) else None)
            _append_contrast(a_name, b_name, value, label_value)
        return results

    if isinstance(raw_contrasts, list):
        for entry in raw_contrasts:
            if not isinstance(entry, dict):
                continue
            if entry.get("enabled", True) is False:
                continue
            value = entry.get("C_AB_exp") or entry.get("target")
            a_raw = entry.get("A") or entry.get("a")
            b_raw = entry.get("B") or entry.get("b")
            if a_raw is None or b_raw is None:
                raise ValueError("Cada contraste debe definir los campos 'A' y 'B'")
            a_name = _normalize_subnet_name(str(a_raw), material)
            b_name = _normalize_subnet_name(str(b_raw), material)
            label = entry.get("type")
            _append_contrast(a_name, b_name, value, str(label) if label else None)
        return results

    return results


@dataclass
class MaterialConfig:
    """Description of the material, subnets, anchors, and optional contrasts."""
=======
@dataclass
class MaterialConfig:
    """Description of the material and available subnets."""
>>>>>>> theirs
=======
@dataclass
class MaterialConfig:
    """Description of the material and available subnets."""
>>>>>>> theirs

    material: str
    subnets: List[str]
    anchors: Dict[str, Dict[str, float]]
<<<<<<< ours
<<<<<<< ours
    primes: Tuple[int, ...] = DEFAULT_PRIMES
    prime_layers: List[int] = field(default_factory=lambda: [1 for _ in PRIMES])
    constraints: ParameterBounds = field(default_factory=ParameterBounds)
    freeze_primes: Tuple[int, ...] = field(default_factory=tuple)
    layers: Dict[str, int] = field(default_factory=dict)
    eta: float = 1.8e-5
    thermal_scales: Dict[str, float] = field(default_factory=dict)
    contrasts: List[ContrastTarget] = field(default_factory=list)
    subnet_configs: Dict[str, SubnetConfig] = field(default_factory=dict)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    @classmethod
    def from_file(cls, path: Path) -> "MaterialConfig":
        data = json.loads(Path(path).read_text())
        material = str(data["material"])
<<<<<<< ours
<<<<<<< ours
        primes, prime_layer_map = _parse_prime_layers(data.get("primes"))
        subnet_configs, subnet_order = _parse_subnet_configs(material, data, prime_layer_map)
        if not subnet_order:
            raise ValueError("material_config.json debe incluir al menos una subred habilitada")

        constraints = ParameterBounds.from_dict(data.get("constraints"))

        raw_freeze = data.get("freeze_primes", [])
        freeze_primes = tuple(sorted({int(p) for p in raw_freeze if isinstance(p, (int, float))}))
        freeze_primes = tuple(p for p in freeze_primes if p in primes)

        anchors: Dict[str, Dict[str, float]] = {}
        raw_anchors = data.get("anchors")
        if isinstance(raw_anchors, dict):
            for subnet, anchor_data in raw_anchors.items():
                normalized = _normalize_subnet_name(str(subnet), material)
                if not isinstance(anchor_data, dict):
                    continue
                entry: Dict[str, float] = {}
                f0_value = anchor_data.get("f0") or anchor_data.get("X_anchor") or anchor_data.get("X")
                x_value = anchor_data.get("X") or anchor_data.get("thermal_X")
                if f0_value is not None:
                    entry["f0"] = float(f0_value)
                if x_value is not None:
                    entry["X"] = float(x_value)
                if entry:
                    anchors[normalized] = entry
        for name, subnet_cfg in subnet_configs.items():
            if subnet_cfg.f0_anchor is not None and name not in anchors:
                value = float(subnet_cfg.f0_anchor)
                anchors[name] = {"f0": value}

        layers_input = data.get("layers") if isinstance(data.get("layers"), dict) else {}
        layers: Dict[str, int] = {}
        for name in subnet_order:
            layer_value: Optional[float] = None
            if isinstance(layers_input, dict) and name in layers_input:
                layer_value = layers_input[name]
            elif name in subnet_configs and subnet_configs[name].l_candidates:
                layer_value = subnet_configs[name].l_candidates[0]
            layers[name] = max(1, int(layer_value)) if isinstance(layer_value, (int, float)) else 1
            if name in subnet_configs:
                subnet_configs[name].layer = layers[name]
                subnet_configs[name].l_candidates = [layers[name]]

        thermal_scales: Dict[str, float] = {}
        for name in subnet_order:
            x_value = anchors.get(name, {}).get("X") if name in anchors else None
            thermal_scales[name] = float(x_value) if isinstance(x_value, (int, float)) else 0.0
            if name in subnet_configs:
                subnet_configs[name].thermal_scale = thermal_scales[name]

        eta_value = data.get("eta")
        if eta_value is None and isinstance(data.get("global_params"), dict):
            eta_value = data.get("global_params").get("eta")
        eta = float(eta_value) if isinstance(eta_value, (int, float)) else 1.8e-5

        contrasts = _parse_contrasts(material, subnet_order, data.get("contrasts"))
        optimization = OptimizationSettings.from_dict(data.get("optimization"))

        return cls(
            material=material,
            subnets=subnet_order,
            anchors=anchors,
            primes=primes,
            prime_layers=[prime_layer_map[prime] for prime in PRIMES],
            constraints=constraints,
            freeze_primes=freeze_primes,
            layers=layers,
            eta=eta,
            thermal_scales=thermal_scales,
            contrasts=contrasts,
            subnet_configs=subnet_configs,
            optimization=optimization,
        )
=======
=======
>>>>>>> theirs
        subnets = [str(name) for name in data["subnets"]]
        anchors: Dict[str, Dict[str, float]] = {}
        raw_anchors = data.get("anchors", {})
        for subnet, anchor_data in raw_anchors.items():
            anchors[subnet] = {key: float(value) for key, value in anchor_data.items()}
        return cls(material=material, subnets=subnets, anchors=anchors)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs


@dataclass
class TargetDataset:
    """Collection of subnet targets and optional contrast entries."""

    subnets: Dict[str, SubnetTarget]
    contrasts: List[ContrastTarget]

    @classmethod
    def from_file(cls, path: Path) -> "TargetDataset":
        raw = json.loads(Path(path).read_text())
        subnets: Dict[str, SubnetTarget] = {}
        contrasts: List[ContrastTarget] = []
        for key, value in raw.items():
            if "_vs_" in key:
                parts = key.split("_vs_")
                if len(parts) != 2:
                    raise ValueError(f"Invalid contrast key: {key}")
                base = parts[0]
                other = parts[1]
                prefix = base.split("_")[0]
                if "_" not in other:
                    other = f"{prefix}_{other}"
                contrast_value = float(value["C_AB_exp"]) if isinstance(value, dict) else float(value)
                contrasts.append(ContrastTarget(subnet_a=base, subnet_b=other, value=contrast_value))
            else:
                if not isinstance(value, dict):
                    raise TypeError(f"Expected object for subnet target '{key}'")
                subnets[key] = SubnetTarget.from_dict(value)
<<<<<<< ours
<<<<<<< ours
        for name, target in subnets.items():
            if target.prime_value is None:
                raise ValueError(f"Target '{name}' is missing 'input_exponents' or valid prime_value")
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        return cls(subnets=subnets, contrasts=contrasts)


def load_loss_weights(path: Optional[Path]) -> LossWeights:
    """Load loss weights from JSON or return defaults when ``path`` is ``None``."""

    if path is None:
        return LossWeights()
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError("Loss weight file must contain a JSON object")
    return LossWeights.from_json(data)  # type: ignore[arg-type]
<<<<<<< ours
<<<<<<< ours
=======

>>>>>>> theirs
=======

>>>>>>> theirs
