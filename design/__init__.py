"""
Design Strategies for Splitting RCT and External Data Pools.

This module provides strategies to determining:
1. The split of RCT data into Treatment and Internal Control.
2. The optimal augmentation size (N_aug) from external data.

Classes:
- FixedRatioDesign: Standard fixed allocation (e.g., 1:1).
- EnergyOptimisedDesign: Optimises N_aug based on energy distance between Target and Augmented Control.
- PooledEnergyMinimizer: Optimises N_aug based on energy distance between Pooled RCT and External data.
"""

from .base import BaseDesign
from .fixed_ratio import FixedRatioDesign
from .energy_optimised import EnergyOptimisedDesign
from .pooled_energy import PooledEnergyMinimizer
from .ipsw_balance import IPSWBalanceDesign

__all__ = [
    "BaseDesign",
    "FixedRatioDesign",
    "EnergyOptimisedDesign",
    "PooledEnergyMinimizer",
    "IPSWBalanceDesign"
]