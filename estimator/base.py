from abc import ABC, abstractmethod
from structures import SplitData, EstimationResult

class BaseEstimator(ABC):
    """
    Abstract base class for estimators.
    """
    def __init__(self, n_external: int = None):
        self.n_external = n_external

    @abstractmethod
    def estimate(self, data: SplitData, n_external: int = None, **kwargs) -> EstimationResult:
        """
        Estimate the treatment effect given the split data.

        Args:
            data (SplitData): Object containing X_treat, X_control_int, X_external, etc.

        Returns:
            EstimationResult: Object containing ATE, weights, and bias metrics.
        """
        pass