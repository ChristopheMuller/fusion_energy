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

    def estimate(self, data: SplitData, n_external: int = None, **kwargs) -> EstimationResult:
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
        X_t = kwargs.get('X_t')
        if X_t is None:
            X_t = torch.as_tensor(data.X_treat, dtype=torch.float32, device=self.device)

        X_c = kwargs.get('X_c')
        if X_c is None:
            X_c = torch.as_tensor(data.X_control_int, dtype=torch.float32, device=self.device)

        X_e = kwargs.get('X_e')
        if X_e is None:
            X_e = torch.as_tensor(data.X_external, dtype=torch.float32, device=self.device)

        # 2. Optimise Soft Weights (Metrics API)
        # Learn soft weights such that X_c + (X_e weighted) approx X_t
        logits = optimise_soft_weights(
            X_source=X_e,
            X_target=X_t,
            X_internal=X_c,
            target_n_aug=target_n,
            lr=self.lr,
            n_iter=self.n_iter,
            dist_st=kwargs.get('dist_st'),
            dist_ss=kwargs.get('dist_ss'),
            dist_is=kwargs.get('dist_is'),
            dist_st_sum=kwargs.get('dist_st_sum'),
            dist_is_sum=kwargs.get('dist_is_sum')
        )
        probs = F.softmax(logits, dim=0)
        
        # 3. Select Best Sample (Stochastic Selection)
        # We pick the specific binary subset that minimizes energy
        # Avoid double-passing X_t, X_c, X_e if they are already in kwargs
        pass_kwargs = {k: v for k, v in kwargs.items() if k not in ['X_t', 'X_c', 'X_e']}
        best_indices, min_energy = self._select_best_sample(X_t, X_c, X_e, probs, target_n, **pass_kwargs)
        
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

    def _select_best_sample(self, X_t, X_c, X_e, probs, n_sampled, **kwargs):
        """Samples k_best subsets and picks the one minimizing Pooled Energy."""
        # A. Sample
        batch_idx_tensor = torch.multinomial(
            probs.expand(self.k_best, -1),
            num_samples=n_sampled,
            replacement=False
        )
        
        dist_st = kwargs.get('dist_st')
        dist_ss = kwargs.get('dist_ss')
        dist_is = kwargs.get('dist_is')
        sum_it = kwargs.get('sum_it')
        sum_ii = kwargs.get('sum_ii')

        dist_st_sum = kwargs.get('dist_st_sum')
        dist_is_sum = kwargs.get('dist_is_sum')

        if dist_st_sum is not None and dist_ss is not None and (X_c is None or (dist_is_sum is not None and sum_it is not None and sum_ii is not None)):
            # Optimized path using precomputed distances
            # 1. Cross Term: Source <-> Target
            # sum_st[k] = sum over s in subset, t in target of d(s, t)
            sum_st = dist_st_sum[batch_idx_tensor].sum(dim=1)

            # 2. Self Term: Source <-> Source
            # We need the sum of the submatrix for each batch
            sum_ss = dist_ss[batch_idx_tensor.unsqueeze(-1), batch_idx_tensor.unsqueeze(1)].sum(dim=(1, 2))

            if X_c is not None:
                n_i = X_c.shape[0]
                n_pool = n_i + n_sampled

                # Cross Internal <-> Source
                sum_is = dist_is_sum[batch_idx_tensor].sum(dim=1)

                term_cross = (sum_it + sum_st) / (n_pool * X_t.shape[0])
                term_self = (sum_ii + sum_ss + 2 * sum_is) / (n_pool**2)
                energies = 2 * term_cross - term_self
            else:
                term_cross = sum_st / (n_sampled * X_t.shape[0])
                term_self = sum_ss / (n_sampled**2)
                energies = 2 * term_cross - term_self
        else:
            # B. Prepare Batches
            X_source_batch = X_e[batch_idx_tensor] # (k, n, dim)

            # C. Compute Energy (Metrics API)
            energies = compute_batch_energy(
                X_source_batch=X_source_batch,
                X_target=X_t,
                X_internal=X_c,
                sum_it=sum_it,
                sum_ii=sum_ii
            )
        
        # D. Pick Best
        best_k_idx = torch.argmin(energies).item()
        min_energy = energies[best_k_idx].item()
        return batch_idx_tensor[best_k_idx].cpu().numpy(), min_energy