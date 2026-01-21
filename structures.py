from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PotentialOutcomes:
    X: np.ndarray
    Y0: np.ndarray
    Y1: Optional[np.ndarray] = None

@dataclass
class SplitData:
    X_treat: np.ndarray
    Y_treat: np.ndarray
    X_control_int: np.ndarray
    Y_control_int: np.ndarray
    X_external: np.ndarray
    Y_external: np.ndarray
    true_sate: float
    target_n_aug: int

@dataclass
class EstimationResult:
    ate_est: float
    bias: float
    error: float
    weights_internal: np.ndarray
    weights_external: np.ndarray

    def n_external_used(self) -> int:
        return np.sum(self.weights_external > 0)