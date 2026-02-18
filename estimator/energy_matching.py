import numpy as np
import torch
import torch.nn.functional as F
import time

from structures import SplitData, EstimationResult
from metrics import optimise_soft_weights, compute_batch_energy
from .base import BaseEstimator

class Energy_MatchingEstimator(BaseEstimator):
    """
    Selects a subset of external controls that minimizes the Energy Distance 
    between the pooled control arm (Internal + Selected External) and the Treatment arm.
    """
    def __init__(self, k_best=300, lr=0.05, n_iter=1000, device=None, n_external: int = None):
        super().__init__(n_external=n_external)
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
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
        X_t = torch.as_tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.as_tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.as_tensor(data.X_external, dtype=torch.float32, device=self.device)

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
        probs = F.softmax(logits, dim=0)
        
        # 3. Select Best Sample (Stochastic Selection)
        # We pick the specific binary subset that minimizes energy
        best_indices, min_energy = self._select_best_sample(X_t, X_c, X_e, probs, target_n)
        
        # 4. Construct Estimation Result
        w_ext = np.zeros(n_ext)
        w_ext[best_indices] = 1.0
        w_int = probs.cpu().numpy()
        
        # ATT Estimate
        y0_weighted = (np.sum(data.Y_control_int) + np.sum(data.Y_external[best_indices])) / (n_int + target_n)
        ate = y1_mean - y0_weighted
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_int,
            weights_external=w_ext,
            energy_distance=min_energy,
            estimation_time=time.time() - start_time
        )

    def _select_best_sample(self, X_t, X_c, X_e, probs, n_sampled):
        """Samples k_best subsets and picks the one minimizing Pooled Energy."""
        # A. Sample
        batch_idx_tensor = torch.multinomial(
            probs.expand(self.k_best, -1),
            num_samples=n_sampled,
            replacement=False
        )
        
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
        min_energy = energies[best_k_idx].item()
        return batch_idx_tensor[best_k_idx].cpu().numpy(), min_energy