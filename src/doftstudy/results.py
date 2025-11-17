"""Common result structures for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass
<<<<<<< ours
<<<<<<< ours
from typing import Dict, Tuple
=======
>>>>>>> theirs
=======
>>>>>>> theirs

from .data import SubnetParameters
from .loss import LossBreakdown
from .model import SimulationResult


@dataclass
class SubnetSimulation:
    """Record the outcome of optimising a subnet."""

    parameters: SubnetParameters
    loss: LossBreakdown
    simulation_result: SimulationResult

<<<<<<< ours
<<<<<<< ours

@dataclass
class SimulationRun:
    """Container for the outcome of a full material run (possibly per seed)."""

    label: str
    seed: int
    primes: Tuple[int, ...]
    freeze_primes: Tuple[int, ...]
    subnet_results: Dict[str, SubnetSimulation]
    base_loss: float
=======
>>>>>>> theirs
=======
>>>>>>> theirs
