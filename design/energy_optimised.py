from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from structures import PotentialOutcomes, SplitData
from metrics import optimise_soft_weights, compute_batch_energy
from .base import BaseDesign

class EnergyOptimisedDesign(BaseDesign):
    """
    Simulates RCT splits to find an optimal 'n_aug' that minimizes the 
    Energy Distance between the RCT Treatment arm and the Augmented Control arm.
    """
    def __init__(self, 
                 k_folds=5, 
                 k_best=50, 
                 n_min=2, 
                 n_max=200, 
                 lr=0.05, 
                 n_iter=300,
                 ratio_trt_after_augmentation=None,
                 ratio_trt_before_augmentation=None,
                 device=None):
        self.k_folds = k_folds
        self.k_best = k_best
        self.n_min = n_min
        self.n_max = n_max
        self.lr = lr
        self.n_iter = n_iter
        self.ratio_trt_after_augmentation = ratio_trt_after_augmentation
        self.ratio_trt_before_augmentation = ratio_trt_before_augmentation

        if self.ratio_trt_before_augmentation is None and self.ratio_trt_after_augmentation is None:
            self.ratio_trt_before_augmentation = 0.5

        if self.ratio_trt_before_augmentation is not None and self.ratio_trt_after_augmentation is not None:
            raise ValueError("Specify only one of ratio_trt_before_augmentation or ratio_trt_after_augmentation.")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.target_n_aug = None

    def _estimate_optimal_augmentation(self, X_rct, X_ext, rng: np.random.Generator):
        n_rct = X_rct.shape[0]
        optimal_ns = []
        X_ext_torch = torch.tensor(X_ext, dtype=torch.float32, device=self.device)
        
        d_ss = torch.cdist(X_ext_torch, X_ext_torch)
        
        for fold in range(self.k_folds):
            indices = rng.permutation(n_rct)
            n_treat_sim = n_rct // 2
            
            X_t_sim = torch.tensor(X_rct[indices[:n_treat_sim]], dtype=torch.float32, device=self.device)
            X_c_sim = torch.tensor(X_rct[indices[n_treat_sim:]], dtype=torch.float32, device=self.device)
            
            logits = self._optimise_soft_weights(X_t_sim, X_c_sim, X_ext_torch, d_ss=d_ss)
            
            # Now using Golden Section Search
            best_n_fold = self._search_best_n(X_t_sim, X_c_sim, X_ext_torch, logits)
            optimal_ns.append(best_n_fold)

        return int(np.median(optimal_ns))

    def _optimise_soft_weights(self, X_t, X_c, X_e, d_ss=None):
        n_e = X_e.shape[0]
        proxy_n = (self.n_min + min(self.n_max, n_e)) // 2
        
        return optimise_soft_weights(
            X_source=X_e,
            X_target=X_t,
            X_internal=X_c,
            target_n_aug=proxy_n,
            lr=self.lr,
            n_iter=self.n_iter,
            dist_ss=d_ss
        )

    def _evaluate_n_energy(self, n, logits, X_t, X_c, X_e):
        n = int(round(n))
        if n <= 0: return 9999.0
        
        probs = F.softmax(logits, dim=0)
        batch_idx = torch.multinomial(
            probs.expand(self.k_best, -1),
            num_samples=n,
            replacement=False
        )
        
        X_aug = X_e[batch_idx] 
        energies = compute_batch_energy(X_aug, X_t, X_c)
        
        return torch.min(energies).item()

    def _search_best_n(self, X_t, X_c, X_e, logits):
        """
        Golden Section Search for finding the minimum of a unimodal function.
        """
        n_e = X_e.shape[0]
        a = self.n_min
        b = min(self.n_max, n_e)
        
        # Golden ratio constant
        inv_phi = (np.sqrt(5) - 1) / 2  # ~0.618
        
        cache = {}
        def get_score(n):
            n_int = int(round(n))
            if n_int in cache: return cache[n_int]
            val = self._evaluate_n_energy(n_int, logits, X_t, X_c, X_e)
            cache[n_int] = val
            return val

        # Initial points
        x1 = b - inv_phi * (b - a)
        x2 = a + inv_phi * (b - a)
        f1 = get_score(x1)
        f2 = get_score(x2)

        # Iterate until the range is small (e.g., 2 units)
        while (b - a) > 2:
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - inv_phi * (b - a)
                f1 = get_score(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + inv_phi * (b - a)
                f2 = get_score(x2)

        # Final check of the small bracket
        candidates = range(int(round(a)), int(round(b)) + 1)
        scores = [get_score(n) for n in candidates]
        return candidates[np.argmin(scores)]

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes, rng: Optional[np.random.Generator] = None) -> SplitData:
        local_rng = rng if rng is not None else np.random.default_rng()
        
        best_n_aug = self._estimate_optimal_augmentation(rct_pool.X, ext_pool.X, local_rng)
        self.target_n_aug = best_n_aug
        n_rct = rct_pool.X.shape[0]
        
        if self.ratio_trt_before_augmentation is not None:
            n_treat = int((n_rct) * self.ratio_trt_before_augmentation)
        else:
            n_treat = int((n_rct + best_n_aug) * self.ratio_trt_after_augmentation)
        
        n_treat = min(max(n_treat, 10), n_rct - 10)

        indices = local_rng.permutation(n_rct)
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