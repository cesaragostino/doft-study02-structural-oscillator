"""Data structures and parsers for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math


PRIMES: Tuple[int, int, int, int] = (2, 3, 5, 7)
PRIME_KEYS: Tuple[str, str, str, str] = ("r2", "r3", "r5", "r7")
DELTA_KEYS: Tuple[str, str, str, str] = ("d2", "d3", "d5", "d7")
DEFAULT_LAMBDA_GEO: Tuple[float, float, float, float] = (1.0, 0.8, 0.4, 0.1)
PrimeVector = Dict[str, float]


@dataclass
class ParameterBounds:
    """Bounds for the search space."""

    ratios: Tuple[float, float] = (0.0, 1.5)
    deltas: Tuple[float, float] = (-0.25, 0.25)
    f0: Tuple[float, float] = (0.2, 50.0)
    delta_T: Tuple[float, float] = (-0.1, 0.1)
    delta_space: Tuple[float, float] = (-0.1, 0.1)

    @classmethod
    def from_cli(cls, override: Optional[Dict[str, Iterable[float]]]) -> "ParameterBounds":
        if not override:
            return cls()

        def _pair(key: str, default: Tuple[float, float]) -> Tuple[float, float]:
            value = override.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                lo, hi = float(value[0]), float(value[1])
                if hi < lo:
                    lo, hi = hi, lo
                if lo == hi:
                    hi = lo + 1e-6
                return (lo, hi)
            return default

        return cls(
            ratios=_pair("ratios_bounds", cls.ratios),
            deltas=_pair("deltas_bounds", cls.deltas),
            f0=_pair("f0_bounds", cls.f0),
            delta_T=_pair("delta_T_bounds", cls.delta_T),
            delta_space=_pair("delta_space_bounds", cls.delta_space),
        )


@dataclass
class LossWeights:
    """Collection of weights for each component of the loss function."""

    w_e: float = 1.0
    w_q: float = 0.5
    w_r: float = 0.25
    w_c: float = 0.3
    w_anchor: float = 0.05

    @classmethod
    def from_json(cls, data: Dict[str, float]) -> "LossWeights":
        kwargs = {k: float(v) for k, v in data.items() if hasattr(cls, k)}
        return cls(**kwargs)


@dataclass
class SubnetTarget:
    """Target observables for a single subnet."""

    e_exp: Optional[List[Optional[float]]] = None
    q_exp: Optional[float] = None
    residual_exp: Optional[float] = None
    input_exponents: Optional[List[int]] = None

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
        if input_exponents is not None:
            input_exponents = [int(v) for v in input_exponents]

        return cls(
            e_exp=_clean_e(data.get("e_exp")),
            q_exp=None if q_value is None else float(q_value),
            residual_exp=None if residual_value is None else float(residual_value),
            input_exponents=input_exponents,
        )


@dataclass
class ContrastTarget:
    """Target contrast between two subnets."""

    subnet_a: str
    subnet_b: str
    value: float


@dataclass
class SubnetParameters:
    """Simulation parameters for a subnet."""

    L: int
    f0: float
    ratios: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    layer_assignment: List[int] = field(default_factory=list)
    delta_T: PrimeVector = field(default_factory=dict)
    delta_space: PrimeVector = field(default_factory=dict)
    lambda_band: float = 1.0
    lambda_geo: PrimeVector = field(default_factory=dict)

    def copy(self) -> "SubnetParameters":
        return SubnetParameters(
            L=self.L,
            f0=self.f0,
            ratios=dict(self.ratios),
            delta=dict(self.delta),
            layer_assignment=list(self.layer_assignment),
            delta_T=dict(self.delta_T),
            delta_space=dict(self.delta_space),
            lambda_band=self.lambda_band,
            lambda_geo=dict(self.lambda_geo),
        )


@dataclass
class MaterialConfig:
    """Description of the material and available subnets."""

    material: str
    subnets: List[str]
    anchors: Dict[str, Dict[str, float]]
    xi: Optional[float] = None
    xi_exp: Dict[str, float] = field(default_factory=dict)
    xi_sign: Dict[str, int] = field(default_factory=dict)
    k_skin: float = 0.0
    delta_T: Dict[str, PrimeVector] = field(default_factory=dict)
    delta_space: Dict[str, PrimeVector] = field(default_factory=dict)
    lambda_band: Dict[str, float] = field(default_factory=dict)
    lambda_geo: PrimeVector = field(default_factory=dict)

    def delta_T_for(self, subnet: str) -> PrimeVector:
        return dict(self.delta_T.get(subnet, {}))

    def delta_space_for(self, subnet: str) -> PrimeVector:
        return dict(self.delta_space.get(subnet, {}))

    def lambda_band_for(self, subnet: str) -> float:
        return float(self.lambda_band.get(subnet, 1.0))

    def lambda_geo_vector(self) -> PrimeVector:
        if self.lambda_geo:
            vec = {str(k): float(v) for k, v in self.lambda_geo.items() if isinstance(v, (int, float))}
        else:
            vec = {str(p): DEFAULT_LAMBDA_GEO[idx] for idx, p in enumerate(PRIMES)}
        for prime in PRIMES:
            vec.setdefault(str(prime), 1.0)
        return vec

    @classmethod
    def from_file(cls, path: Path) -> "MaterialConfig":
        data = json.loads(Path(path).read_text())
        material = str(data["material"])
        subnets = [str(name) for name in data["subnets"]]
        anchors: Dict[str, Dict[str, float]] = {}
        raw_anchors = data.get("anchors", {})
        for subnet, anchor_data in raw_anchors.items():
            anchors[subnet] = {key: float(value) for key, value in anchor_data.items()}
        xi_value = data.get("xi")
        if isinstance(xi_value, (int, float)) and math.isfinite(xi_value):
            xi = float(xi_value)
        else:
            xi = None
        xi_exp = {}
        raw_xi_exp = data.get("xi_exp", {})
        if isinstance(raw_xi_exp, dict):
            for key, val in raw_xi_exp.items():
                if isinstance(val, (int, float)) and math.isfinite(val):
                    xi_exp[str(key)] = float(val)
        xi_sign = {}
        raw_sign = data.get("xi_sign", {})
        if isinstance(raw_sign, dict):
            for key, val in raw_sign.items():
                try:
                    xi_sign[str(key)] = int(val)
                except Exception:
                    continue
        k_skin = 0.0
        if isinstance(data.get("k_skin"), (int, float)):
            k_skin = float(data["k_skin"])

        def _clean_prime_vector(raw: object, fallback: PrimeVector) -> PrimeVector:
            vector = {str(p): fallback.get(str(p), 0.0) for p in PRIMES}
            if raw is None:
                return vector
            if isinstance(raw, (int, float)) and math.isfinite(raw):
                return {str(p): float(raw) for p in PRIMES}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    key = str(k)
                    if key not in vector:
                        continue
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        vector[key] = float(v)
            return vector

        def _parse_vector_by_subnet(raw: object, default_value: float = 0.0) -> Dict[str, PrimeVector]:
            base_zero = {str(p): default_value for p in PRIMES}
            out: Dict[str, PrimeVector] = {subnet: dict(base_zero) for subnet in subnets}
            if raw is None:
                return out
            if isinstance(raw, (int, float)):
                shared = _clean_prime_vector(raw, base_zero)
                return {subnet: dict(shared) for subnet in subnets}
            if isinstance(raw, dict):
                has_subnet_keys = any(str(k) in subnets for k in raw.keys())
                if has_subnet_keys:
                    for subnet in subnets:
                        if subnet in raw:
                            out[subnet] = _clean_prime_vector(raw[subnet], base_zero)
                    return out
                shared = _clean_prime_vector(raw, base_zero)
                return {subnet: dict(shared) for subnet in subnets}
            return out

        def _infer_lambda_band(subnet: str) -> float:
            lower = subnet.lower()
            if "pi" in lower:
                return 0.8
            if "sigma" in lower:
                return 1.0
            return 1.0

        raw_lambda_band = data.get("lambda_band")
        lambda_band_map = {subnet: _infer_lambda_band(subnet) for subnet in subnets}
        if isinstance(raw_lambda_band, (int, float)) and math.isfinite(raw_lambda_band):
            lambda_band_map = {subnet: float(raw_lambda_band) for subnet in subnets}
        elif isinstance(raw_lambda_band, dict):
            for subnet in subnets:
                value = raw_lambda_band.get(subnet)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    lambda_band_map[subnet] = float(value)

        lambda_geo = _clean_prime_vector(data.get("lambda_geo"), {str(p): DEFAULT_LAMBDA_GEO[idx] for idx, p in enumerate(PRIMES)})
        delta_T = _parse_vector_by_subnet(data.get("delta_T"), 0.0)
        delta_space = _parse_vector_by_subnet(data.get("delta_space"), 0.0)
        return cls(
            material=material,
            subnets=subnets,
            anchors=anchors,
            xi=xi,
            xi_exp=xi_exp,
            xi_sign=xi_sign,
            k_skin=k_skin,
            delta_T=delta_T,
            delta_space=delta_space,
            lambda_band=lambda_band_map,
            lambda_geo=lambda_geo,
        )


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
        return cls(subnets=subnets, contrasts=contrasts)


def load_loss_weights(path: Optional[Path]) -> LossWeights:
    """Load loss weights from JSON or return defaults when ``path`` is ``None``."""

    if path is None:
        return LossWeights()
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError("Loss weight file must contain a JSON object")
    return LossWeights.from_json(data)  # type: ignore[arg-type]
