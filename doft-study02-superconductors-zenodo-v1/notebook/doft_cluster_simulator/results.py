"""Common result structures for the DOFT cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .data import SubnetParameters
from .loss import LossBreakdown
from .model import SimulationResult


@dataclass
class SubnetSimulation:
    """Record the outcome of optimising a subnet."""

    parameters: SubnetParameters
    loss: LossBreakdown
    simulation_result: SimulationResult

