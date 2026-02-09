"""
Estimator classes for treatment effect estimation.

This module contains methods to estimate the Average Treatment Effect (ATE)
or Average Treatment Effect on the Treated (ATT) given a split dataset.

Classes:
- EnergyMatchingEstimator: Matches external controls using Energy Distance minimization.
- DummyMatchingEstimator: Randomly selects external controls (Baseline).
"""

from .base import BaseEstimator
from .energy_matching import EnergyMatchingEstimator
from .dummy import DummyMatchingEstimator
from .energy_weighting import EnergyWeightingEstimator
from .optimal_energy_matching import OptimalEnergyMatchingEstimator
from .ipw import IPWEstimator
from .energy_pooled_weighting import EnergyPooledWeightingEstimator
from .prognostic_energy_weighting import PrognosticEnergyWeightingEstimator

__all__ = [
    "BaseEstimator",
    "EnergyMatchingEstimator",
    "DummyMatchingEstimator",
    "EnergyWeightingEstimator",
    "OptimalEnergyMatchingEstimator",
    "IPWEstimator",
    "EnergyPooledWeightingEstimator",
    "PrognosticEnergyWeightingEstimator",
]
