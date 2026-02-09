import numpy as np
import time
from structures import SplitData, EstimationResult
from .base import BaseEstimator

class DummyMatchingEstimator(BaseEstimator):
    """
    Randomly selects a subset of external controls.
    Serves as a baseline (dummy) estimator.
    
    If target_n is 0, it uses only internal controls.
    If target_n equals the number of external units, it uses all of them.
    Otherwise, it samples target_n units uniformly at random without replacement.
    """
    def __init__(self, n_external: int = None, seed: int = None):
        super().__init__(n_external=n_external)
        self.seed = seed

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
        rng = np.random.default_rng(self.seed)
        
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        
        # Determine target_n
        if n_external is not None:
            target_n = n_external
        elif self.n_external is not None:
            target_n = int(self.n_external)
        else:
            target_n = data.target_n_aug

        y1_mean = np.mean(data.Y_treat)

        # Case 1: No external augmentation
        if target_n == 0 or n_ext == 0:
            ate = y1_mean - np.mean(data.Y_control_int)
            return EstimationResult(
                ate_est=ate,
                bias=ate - data.true_sate,
                weights_continuous=np.ones(n_ext),
                weights_external=np.zeros(n_ext),
                estimation_time=time.time() - start_time
            )
        
        # Case 2: Select Random Subset
        # If target_n >= n_ext, we take all
        if target_n >= n_ext:
            selected_indices = np.arange(n_ext)
        else:
            selected_indices = rng.choice(n_ext, size=target_n, replace=False)
            
        w_ext = np.zeros(n_ext)
        w_ext[selected_indices] = 1.0
        w_int = np.ones(n_ext) / target_n
        
        # Calculate Outcome
        # We assume simple pooling of Internal Control + Selected External
        # Mean of (Y_control_int + Y_external_selected)
        
        sum_y_control = np.sum(data.Y_control_int) + np.sum(data.Y_external[selected_indices])
        n_control_total = n_int + len(selected_indices)
        
        y0_pooled = sum_y_control / n_control_total
        ate = y1_mean - y0_pooled
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_int,
            weights_external=w_ext,
            estimation_time=time.time() - start_time
        )
