from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from structures import SplitData, EstimationResult

class BaseEstimator(ABC):
    @abstractmethod
    def estimate(self, data: SplitData) -> EstimationResult:
        pass

class IPWEstimator(BaseEstimator):
    def estimate(self, data: SplitData) -> EstimationResult:
        # Standard IPW implementation placeholder
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        
        w_int = np.ones(n_int)
        w_ext = np.ones(n_ext) * (data.target_n_aug / n_ext)
        
        y1_mean = np.mean(data.Y_treat)
        y0_w_sum = np.sum(data.Y_control_int * w_int) + np.sum(data.Y_external * w_ext)
        w_sum = np.sum(w_int) + np.sum(w_ext)
        
        ate = y1_mean - (y0_w_sum / w_sum)
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            error=(ate - data.true_sate)**2,
            weights_internal=w_int,
            weights_external=w_ext
        )

class EnergyMatchingEstimator(BaseEstimator):
    def estimate(self, data: SplitData) -> EstimationResult:
        # Constraint: Weights must be binary and sum to data.target_n_aug
        X_target = data.X_treat # Or Pooled Control based on preference
        X_ext = data.X_external
        target_n = data.target_n_aug
        
        # Simple Greedy selection for demonstration
        # In reality: Minimize Energy(X_target, X_ext[subset])
        dists = np.linalg.norm(X_ext[:, None] - np.mean(X_target, axis=0), axis=2).flatten()
        best_indices = np.argsort(dists)[:target_n]
        
        w_ext = np.zeros(X_ext.shape[0])
        w_ext[best_indices] = 1.0
        w_int = np.ones(data.X_control_int.shape[0])
        
        y1_mean = np.mean(data.Y_treat)
        y0_weighted = (np.sum(data.Y_control_int) + np.sum(data.Y_external[best_indices])) / (np.sum(w_int) + target_n)
        
        ate = y1_mean - y0_weighted
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            error=(ate - data.true_sate)**2,
            weights_internal=w_int,
            weights_external=w_ext
        )