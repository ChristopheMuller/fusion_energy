from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from structures import SplitData, EstimationResult
from sklearn.linear_model import LogisticRegression
from metrics import optimize_soft_weights, compute_batch_energy

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
    def __init__(self, k_best=100, lr=0.05, n_iter=300, device=None):
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def estimate(self, data: SplitData) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        target_n = data.target_n_aug
        y1_mean = np.mean(data.Y_treat)

        # Edge case: No augmentation
        if target_n == 0 or n_ext == 0:
            ate = y1_mean - np.mean(data.Y_control_int)
            return EstimationResult(
                ate_est=ate,
                bias=ate - data.true_sate,
                error=(ate - data.true_sate)**2,
                weights_internal=np.ones(n_int),
                weights_external=np.zeros(n_ext)
            )
        
        # 1. Tensor Setup
        X_t = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)

        # 2. Optimize Soft Weights (Metrics API)
        # We want X_c + X_e_subset approx X_t
        logits = optimize_soft_weights(
            X_source=X_e,
            X_target=X_t,
            X_internal=X_c,
            target_n_aug=target_n,
            lr=self.lr,
            n_iter=self.n_iter
        )
        probs = F.softmax(logits, dim=0).cpu().numpy()
        probs /= probs.sum()
        
        # 3. Select Best Sample (Stochastic Selection)
        best_indices = self._select_best_sample(X_t, X_c, X_e, probs, target_n)
        
        # 4. Construct Estimation Result
        w_ext = np.zeros(n_ext)
        w_ext[best_indices] = 1.0
        
        # ATT Estimate
        y0_weighted = (np.sum(data.Y_control_int) + np.sum(data.Y_external[best_indices])) / (n_int + target_n)
        ate = y1_mean - y0_weighted
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            error=(ate - data.true_sate)**2,
            weights_internal=probs,
            weights_external=w_ext
        )

    def _select_best_sample(self, X_t, X_c, X_e, probs, n_sampled):
        """Samples k_best subsets and picks the one minimizing Pooled Energy."""
        # A. Sample
        pool_idx = np.arange(len(probs))
        
        batch_indices = [
            np.random.choice(pool_idx, size=n_sampled, replace=False, p=probs)
            for _ in range(self.k_best)
        ]
        batch_idx_tensor = torch.tensor(np.array(batch_indices), device=self.device, dtype=torch.long)
        
        # B. Prepare Batches
        X_source_batch = X_e[batch_idx_tensor] # (k, n, dim)
        
        # C. Compute Energy (Metrics API)
        energies = compute_batch_energy(
            X_source_batch=X_source_batch,
            X_target=X_t,
            X_internal=X_c
        )
        
        # D. Pick Best
        best_k_idx = torch.argmin(energies).item()
        return batch_indices[best_k_idx]