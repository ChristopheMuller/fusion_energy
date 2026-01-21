from abc import ABC, abstractmethod
from structures import SplitData, EstimationResult

class BaseEstimator(ABC):
    """
    Abstract base class for estimators.
    """
    @abstractmethod
    def estimate(self, data: SplitData) -> EstimationResult:
        """
        Estimate the treatment effect given the split data.

        Args:
            data (SplitData): Object containing X_treat, X_control_int, X_external, etc.

        Returns:
            EstimationResult: Object containing ATE, weights, and bias metrics.
        """
        pass