import numpy as np
import torch
import torch.nn.functional as F

from structures import PotentialOutcomes, SplitData
from metrics import optimise_soft_weights, compute_batch_energy
from .base import BaseDesign

class PooledEnergyMinimizer(BaseDesign):
    """
    For n = 0, ..., N_ext, finds the n that minimizes Energy distance between:
    - Pooled RCT (Treatment + Control)
    - Best matching population from External data of size n.
    """
    def __init__(self, 
                 k_best=50, 
                 n_min=10, 
                 n_max=500, 
                 lr=0.05, 
                 n_iter=500,
                 device=None):
        self.k_best = k_best
        self.n_min = n_min
        self.n_max = n_max
        self.lr = lr
        self.n_iter = n_iter
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.target_n_aug = None

    def _optimise_soft_weights(self, X_pool, X_ext):
        X_pool_torch = torch.tensor(X_pool, dtype=torch.float32, device=self.device)
        X_e_torch = torch.tensor(X_ext, dtype=torch.float32, device=self.device)
        
        return optimise_soft_weights(
            X_source=X_e_torch,
            X_target=X_pool_torch,
            X_internal=None, 
            target_n_aug=None,
            lr=self.lr,
            n_iter=self.n_iter
        )

    def _evaluate_n_energy(self, n, logits, X_pool, X_ext):
        if n == 0: return 9999.0
        
        X_pool_torch = torch.tensor(X_pool, dtype=torch.float32, device=self.device)
        X_ext_torch = torch.tensor(X_ext, dtype=torch.float32, device=self.device)
        
        # 1. Sample Indices
        probs = F.softmax(logits, dim=0).cpu().numpy()
        probs /= probs.sum()
        pool_idx = np.arange(len(probs))
        
        batch_indices = [
            np.random.choice(pool_idx, size=n, replace=False, p=probs)
            for _ in range(self.k_best)
        ]
        batch_idx = torch.tensor(np.array(batch_indices), device=self.device, dtype=torch.long)
        
        # 2. Compute Energy
        X_cand_ext = X_ext_torch[batch_idx]
        energies = compute_batch_energy(X_cand_ext, X_pool_torch, X_internal=None)
        
        return torch.min(energies).item()

    def _search_best_n(self, X_pool, X_ext, logits):
        left = self.n_min
        right = min(self.n_max, X_ext.shape[0])
        cache = {}
        
        def get_score(n):
            if n in cache: return cache[n]
            val = self._evaluate_n_energy(n, logits, X_pool, X_ext)
            cache[n] = val
            return val
        
        while right - left > 2:
            m1 = left + (right - left) // 3
            m2 = right - (right - left) // 3
            if get_score(m1) < get_score(m2):
                right = m2
            else:
                left = m1
                
        candidates = range(left, right + 1)
        scores = [get_score(n) for n in candidates]
        return candidates[np.argmin(scores)]

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        n_rct = rct_pool.X.shape[0]
        
        # 1. Optimise Weights & Find N (using Pooled RCT)
        logits = self._optimise_soft_weights(rct_pool.X, ext_pool.X)
        best_n_aug = self._search_best_n(rct_pool.X, ext_pool.X, logits)
        self.target_n_aug = best_n_aug
        
        # 2. Perform Split (Balance final sample sizes)
        n_treat = int((n_rct + best_n_aug) / 2)
        n_treat = min(max(n_treat, 10), n_rct - 10) 
        
        indices = np.random.permutation(n_rct)
        idx_t = indices[:n_treat]
        idx_c = indices[n_treat:]
        
        true_sate = np.mean(rct_pool.Y1[indices] - rct_pool.Y0[indices])

        return SplitData(
            X_treat=rct_pool.X[idx_t],
            Y_treat=rct_pool.Y1[idx_t],
            X_control_int=rct_pool.X[idx_c],
            Y_control_int=rct_pool.Y0[idx_c],
            X_external=ext_pool.X,
            Y_external=ext_pool.Y0,
            true_sate=true_sate,
            target_n_aug=best_n_aug
        )