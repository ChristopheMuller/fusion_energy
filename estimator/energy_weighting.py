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
    - Internal units: Weighted by (1 - alpha)
    - External units: Weighted by alpha * probs
    - alpha: Learned mixture parameter in [0, 1]
    - probs: Learned distribution over external units (sums to 1)
    """
    def __init__(self, lr=0.05, n_iter=300, device=None):
        super().__init__()
        self.lr = lr
        self.n_iter = n_iter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        
        y1_mean = np.mean(data.Y_treat)
        
        # 1. Tensor Setup
        X_t = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)

        # 2. Optimization setup
        logits_ext = torch.zeros(n_ext, requires_grad=True, device=self.device)
        # alpha controls the relative weight of external vs internal
        # we start with some initial alpha, e.g. 0.5 (logit 0)
        logit_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        
        opt = torch.optim.Adam([logits_ext, logit_alpha], lr=self.lr)

        # Precompute distances
        d_et = torch.cdist(X_e, X_t)
        sum_et_per_ext = d_et.sum(dim=1)
        
        d_ee = torch.cdist(X_e, X_e)
        
        d_it = torch.cdist(X_c, X_t)
        sum_it = d_it.sum()
        
        d_ii = torch.cdist(X_c, X_c)
        sum_ii = d_ii.sum()
        
        d_ie = torch.cdist(X_c, X_e)
        sum_ie_per_ext = d_ie.sum(dim=0)

        n_t = X_t.shape[0]

        prev_loss = float('inf')
        for _ in range(self.n_iter):
            opt.zero_grad()
            
            w_ext_probs = F.softmax(logits_ext, dim=0)
            alpha = torch.sigmoid(logit_alpha)
            
            # Weighted Source Distribution:
            # P = (1-alpha) * Internal + alpha * External_Weighted
            # Weights: w_i = (1-alpha)/n_int, w_e = alpha * w_ext_probs
            
            # Cross Term: 2 * E[d(Source, Target)]
            # term_it = (1-alpha) * sum_it / (n_int * n_t)
            # term_et = alpha * dot(w_ext_probs, sum_et_per_ext) / n_t
            
            term_cross = (1-alpha) * sum_it / (n_int * n_t) + alpha * torch.dot(w_ext_probs, sum_et_per_ext) / n_t
            
            # Self Term: E[d(Source, Source)]
            # Term II: (1-alpha)^2 * sum_ii / (n_int^2)
            # Term IE: 2 * (1-alpha) * alpha * dot(w_ext_probs, sum_ie_per_ext) / n_int
            # Term EE: alpha^2 * dot(w_ext_probs, mv(d_ee, w_ext_probs))
            
            term_self = ( (1-alpha)**2 * sum_ii / (n_int**2) + 
                          2 * (1-alpha) * alpha * torch.dot(w_ext_probs, sum_ie_per_ext) / n_int +
                          alpha**2 * torch.dot(w_ext_probs, torch.mv(d_ee, w_ext_probs)) )
            
            loss = 2 * term_cross - term_self
            
            loss.backward()
            opt.step()
            
            if abs(prev_loss - loss.item()) < 1e-6:
                break
            prev_loss = loss.item()

        # 3. Convert to Weights for EstimationResult
        # We need weights that sum to something meaningful.
        # Let's use the natural scaling where internal weights are 1.0 (after normalization)
        # Final weights for external: alpha / (1 - alpha) * n_int * w_ext_probs
        
        final_alpha = torch.sigmoid(logit_alpha).item()
        final_probs = F.softmax(logits_ext, dim=0).detach().cpu().numpy()
        
        # Scaling such that internal units have weight 1.0
        # Source = sum(w_i * delta_i) + sum(w_e * delta_e)
        # where sum(w_i) = 1-alpha, sum(w_e) = alpha
        # To have w_i = 1/n_int, we multiply by n_int / (1-alpha)
        
        scale = n_int / (1 - final_alpha) if final_alpha < 1.0 else 1e6
        w_ext = final_probs * (final_alpha * scale)
        
        # y0_mean = (sum(Y_int * 1.0) + sum(Y_ext * w_ext)) / (n_int + sum(w_ext))
        y0_weighted_sum = np.sum(data.Y_control_int) + np.sum(data.Y_external * w_ext)
        total_weight = n_int + np.sum(w_ext)
        y0_weighted_mean = y0_weighted_sum / total_weight
        
        ate = y1_mean - y0_weighted_mean
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_ext,
            weights_external=w_ext,
            energy_distance=prev_loss
        )