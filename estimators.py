from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from structures import SplitData, EstimationResult
from sklearn.linear_model import LogisticRegression
from metrics import optimize_soft_weights

"""
Estimator classes for treatment effect estimation.

Previous steps include:
1. Data generation of (pooled) RCT data and external (control) data.
2. RCT splitting into treated and internal control groups.

This module takes the split data and applies different estimation strategies:
*WEIGHTNG*
- IPWEstimator: Inverse Probability Weighting to reweight external controls.
- EnergyWeigthingEstimator: ...

*MATCHING*
- EnergyMatchingEstimator: Uses energy distance to select external controls.

*DUMMY*
- ...

"""

class BaseEstimator(ABC):
    @abstractmethod
    def estimate(self, data: SplitData) -> EstimationResult:
        pass

class IPWEstimator(BaseEstimator):
    def estimate(self, data: SplitData) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        
        y1_mean = np.mean(data.Y_treat)

        if data.target_n_aug == 0 or n_ext == 0:
            y0_mean = np.mean(data.Y_control_int)
            ate = y1_mean - y0_mean
            return EstimationResult(
                ate_est=ate,
                bias=ate - data.true_sate,
                error=(ate - data.true_sate)**2,
                weights_internal=np.ones(n_int),
                weights_external=np.zeros(n_ext)
            )

        X_pool = np.vstack([data.X_control_int, data.X_external])
        source = np.array([1] * n_int + [0] * n_ext)
        
        prop_model = LogisticRegression(l1_ratio=0, solver='lbfgs')
        prop_model.fit(X_pool, source)
        
        p_int = prop_model.predict_proba(X_pool)[:, 1]
        
        eps = 1e-6
        p_int = np.clip(p_int, eps, 1 - eps)
        
        weights = p_int / (1 - p_int)
        
        w_int = np.ones(n_int) 
        w_ext = weights[n_int:]

        y0_w_sum = np.sum(data.Y_control_int * w_int) + np.sum(data.Y_external * w_ext)
        w_sum = np.sum(w_int) + np.sum(w_ext)
        
        if w_sum == 0:
            ate = y1_mean # Or handle as error
        else:
            ate = y1_mean - (y0_w_sum / w_sum)
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            error=(ate - data.true_sate)**2,
            weights_internal=w_int,
            weights_external=w_ext
        )

class EnergyMatchingEstimator(BaseEstimator):
    def __init__(self, lr=0.05, n_iter=300, device=None):
        self.lr = lr
        self.n_iter = n_iter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _optimize_soft_weights(self, X_t, X_c, X_e, target_n_aug):
        return optimize_soft_weights(
            X_source=X_e,
            X_target=X_t,
            X_internal=X_c,
            target_n_aug=target_n_aug,
            lr=self.lr,
            n_iter=self.n_iter
        )

    def estimate(self, data: SplitData) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        target_n = data.target_n_aug

        y1_mean = np.mean(data.Y_treat)

        if target_n == 0 or n_ext == 0:
            y0_mean = np.mean(data.Y_control_int)
            ate = y1_mean - y0_mean
            return EstimationResult(
                ate_est=ate,
                bias=ate - data.true_sate,
                error=(ate - data.true_sate)**2,
                weights_internal=np.ones(n_int),
                weights_external=np.zeros(n_ext)
            )
        
        X_t_torch = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c_torch = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e_torch = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)

        logits = self._optimize_soft_weights(X_t_torch, X_c_torch, X_e_torch, target_n)
        
        best_indices = torch.topk(logits, k=min(target_n, n_ext)).indices.cpu().numpy()
        
        w_ext = np.zeros(n_ext)
        w_ext[best_indices] = 1.0
        w_int = np.ones(n_int)
        
        y0_control_int = data.Y_control_int
        y0_external_selected = data.Y_external[best_indices]
        
        y0_weighted = (np.sum(y0_control_int) + np.sum(y0_external_selected)) / (n_int + len(best_indices))
        
        ate = y1_mean - y0_weighted
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            error=(ate - data.true_sate)**2,
            weights_internal=w_int,
            weights_external=w_ext
        )
