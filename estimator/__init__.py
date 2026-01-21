"""
Estimator classes for treatment effect estimation.

This module contains methods to estimate the Average Treatment Effect (ATE)
or Average Treatment Effect on the Treated (ATT) given a split dataset.

Classes:
- IPWEstimator: Inverse Probability Weighting (using Logistic Regression).
- EnergyMatchingEstimator: Matches external controls using Energy Distance minimization.
- DummyMatchingEstimator: Randomly selects external controls (Baseline).
"""

from .base import BaseEstimator
from .ipw import IPWEstimator
from .energy_matching import EnergyMatchingEstimator
from .dummy import DummyMatchingEstimator

__all__ = [
    "BaseEstimator",
    "IPWEstimator",
    "EnergyMatchingEstimator",
    "DummyMatchingEstimator"
]
