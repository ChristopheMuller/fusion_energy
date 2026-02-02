import numpy as np
import torch
import torch.nn.functional as F

from structures import SplitData, EstimationResult
from metrics import optimise_soft_weights
from .base import BaseEstimator

class EnergyWeightingEstimator(BaseEstimator):
    """
    Estimates ATE by weighting the external control arm.
    
    The weights for the external units are optimized to minimize the Energy Distance 
    between the pooled control arm (Internal + Weighted External) and the Treatment arm.
    
    Weights configuration:
    - Internal units: Fixed weight = 1.0
    - External units: Optimized weights scaled to sum to 'target_n' (n_external)
    """
    def __init__(self, lr=0.05, n_iter=300, device=None, n_external: int = None):
        super().__init__(n_external=n_external)
        self.lr = lr
        self.n_iter = n_iter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        
        # Determine target_n (Effective sample size for external data)
        if n_external is not None:
            target_n = n_external
        elif self.n_external is not None:
            target_n = int(self.n_external)
        else:
            target_n = data.target_n_aug

        y1_mean = np.mean(data.Y_treat)
        
        # 1. Tensor Setup
        X_t = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)

        # 2. Optimize Soft Weights (Metrics API)
        logits = optimise_soft_weights(
            X_source=X_e,
            X_target=X_t,
            X_internal=X_c,
            target_n_aug=target_n,
            lr=self.lr,
            n_iter=self.n_iter
        )
        
        # 3. Convert Logits to Weights
        probs = F.softmax(logits, dim=0).cpu().numpy()
        w_ext = probs * target_n
        
        y0_weighted_sum = np.sum(data.Y_control_int) + np.sum(data.Y_external * w_ext)
        total_control_weight = n_int + target_n
        
        y0_weighted_mean = y0_weighted_sum / total_control_weight
        
        ate = y1_mean - y0_weighted_mean
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_ext,
            weights_external=w_ext
        )