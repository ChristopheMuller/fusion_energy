import numpy as np
import torch
import torch.nn.functional as F

from structures import SplitData, EstimationResult
from metrics import optimise_soft_weights, compute_batch_energy
from .base import BaseEstimator

class EnergyMatchingEstimator(BaseEstimator):
    """
    Selects a subset of external controls that minimizes the Energy Distance 
    between the pooled control arm (Internal + Selected External) and the Treatment arm.
    """
    def __init__(self, k_best=100, lr=0.05, n_iter=300, device=None, n_external: int = None):
        super().__init__(n_external=n_external)
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
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
        
        # 1. Tensor Setup
        X_t = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)

        # 2. Optimise Soft Weights (Metrics API)
        # Learn soft weights such that X_c + (X_e weighted) approx X_t
        logits = optimise_soft_weights(
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
        # We pick the specific binary subset that minimizes energy
        best_indices = self._select_best_sample(X_t, X_c, X_e, probs, target_n)
        
        # 4. Construct Estimation Result
        w_ext = np.zeros(n_ext)
        w_ext[best_indices] = 1.0
        w_int = np.ones(n_int)
        
        # ATT Estimate
        y0_weighted = (np.sum(data.Y_control_int) + np.sum(data.Y_external[best_indices])) / (n_int + target_n)
        ate = y1_mean - y0_weighted
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_internal=w_int,
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