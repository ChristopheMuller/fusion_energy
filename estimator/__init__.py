"""
Estimator classes for treatment effect estimation.

This module contains methods to estimate the Average Treatment Effect (ATE)
or Average Treatment Effect on the Treated (ATT) given a split dataset.
"""

from .base import BaseEstimator
from .dummy import Dummy_MatchingEstimator

from .energy_weighting import Energy_WeightingEstimator
from .energy_matching import Energy_MatchingEstimator
from .optimal_energy_matching import Optimal_Energy_MatchingEstimator
from .optimal_energy_matching_prog import OptimalProg_Energy_MatchingEstimator

from .energyProg_weighting import EnergyProg_WeightingEstimator
from .energyProg_matching import EnergyProg_MatchingEstimator

from .MSEProg_matching import MSEProg_MatchingEstimator
from .optimal_MSEProg_matching import Optimal_MSEProg_MatchingEstimator

from .r_bridge import REstimator

__all__ = [
    "BaseEstimator",
    "Dummy_MatchingEstimator",

    "Energy_WeightingEstimator",
    "Energy_MatchingEstimator",
    "Optimal_Energy_MatchingEstimator",
    "OptimalProg_Energy_MatchingEstimator",

    "EnergyProg_WeightingEstimator",
    "EnergyProg_MatchingEstimator",

    "MSEProg_MatchingEstimator",
    "Optimal_MSEProg_MatchingEstimator",

    "REstimator",
]
