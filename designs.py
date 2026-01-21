from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from structures import PotentialOutcomes, SplitData
from metrics import optimize_soft_weights, compute_batch_energy

"""
Design Strategies for Splitting RCT and External Data Pools.

Previous steps generated (pooled) RCT data with potential outcomes, and external data with only control outcomes.

Here, the simulation proceeds as follows:
- Use a Design strategy to split the RCT data into treatment and internal control groups,
  and (potentially) determine how many external units to augment with (as the split may depend on this).
- Return a SplitData object containing all necessary subsets for estimation.
- Note: this step does NOT observe outcomes for RCT units as one does not yet know which units are treated vs control.

Next steps will involve applying Estimators on the SplitData to estimate treatment effects.
"""

class BaseDesign(ABC):
    @abstractmethod
    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        pass

class FixedRatioDesign(BaseDesign):
    def __init__(self, treat_ratio=0.5, fixed_n_aug=100):
        self.treat_ratio = treat_ratio
        self.fixed_n_aug = fixed_n_aug

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        n_rct = rct_pool.X.shape[0]
        n_treat = int(n_rct * self.treat_ratio)
        
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
            target_n_aug=self.fixed_n_aug
        )

class EnergyOptimizedDesign(BaseDesign):
    def __init__(self, 
                 k_folds=5, 
                 k_best=50, 
                 n_min=5, 
                 n_max=200, 
                 lr=0.05, 
                 n_iter=300,
                 ratio_trt_after_augmentation=0.5,
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

        if self.ratio_trt_before_augmentation is not None and self.ratio_trt_after_augmentation is not None:
            raise ValueError("Specify only one of ratio_trt_before_augmentation or ratio_trt_after_augmentation.")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _estimate_optimal_augmentation(self, X_rct, X_ext):
        # Data preparation
        n_rct = X_rct.shape[0]
        optimal_ns = []

        # Convert to tensors once if possible, but we split X_rct inside loop
        X_ext_torch = torch.tensor(X_ext, dtype=torch.float32, device=self.device)
        
        for fold in range(self.k_folds):
            # 1. Split RCT 1:1 into "Simulated Target" (Treat) and "Simulated Internal" (Control)
            indices = np.random.permutation(n_rct)
            n_treat_sim = n_rct // 2
            
            X_t_sim = torch.tensor(X_rct[indices[:n_treat_sim]], dtype=torch.float32, device=self.device)
            X_c_sim = torch.tensor(X_rct[indices[n_treat_sim:]], dtype=torch.float32, device=self.device)
            
            # 2. Get the "Quality Scores" (logits) for External Units via Gradient Descent
            # This runs ONCE per fold to rank external units.
            logits = self._optimize_soft_weights(X_t_sim, X_c_sim, X_ext_torch)
            
            # 3. Find optimal n using Ternary Search (exploiting convexity)
            # We use the logits to sample "good" sets for every n we test.
            best_n_fold = self._search_best_n(X_t_sim, X_c_sim, X_ext_torch, logits)
            optimal_ns.append(best_n_fold)

        return int(np.mean(optimal_ns))

    def _optimize_soft_weights(self, X_t, X_c, X_e):
        """
        Runs Gradient Descent to find a soft probability mask over X_e 
        that minimizes Energy(Pooled, Target).
        """
        n_e = X_e.shape[0]
        proxy_n = (self.n_min + min(self.n_max, n_e)) // 2
        
        return optimize_soft_weights(
            X_source=X_e,
            X_target=X_t,
            X_internal=X_c,
            target_n_aug=proxy_n,
            lr=self.lr,
            n_iter=self.n_iter
        )

    def _evaluate_n_energy(self, n, logits, X_t, X_c, X_e):
        """
        Samples 'k_best' subsets of size 'n' using 'logits',
        computes the EXACT Energy Distance for the pooled set,
        returns the minimum energy found.
        """
        if n == 0: return 9999.0 # Edge case
        
        # 1. Sample Indices
        probs = F.softmax(logits, dim=0)
        # Using numpy for choice is often easier/safer for "replace=False" across batches
        probs_np = probs.cpu().numpy()
        probs_np /= probs_np.sum()
        
        pool_idx = np.arange(len(probs_np))
        
        # Generate k_best masks
        batch_indices_list = [
            np.random.choice(pool_idx, size=n, replace=False, p=probs_np)
            for _ in range(self.k_best)
        ]
        
        # Stack into tensor: (k_best, n)
        batch_idx = torch.tensor(np.array(batch_indices_list), device=self.device, dtype=torch.long)
        
        # 2. Gather Data: (k_best, n, dim)
        X_aug = X_e[batch_idx] 
        
        # 3. Compute Energy Efficiently (Batch Mode)
        energies = compute_batch_energy(X_aug, X_t, X_c)
        
        return torch.min(energies).item()

    def _search_best_n(self, X_t, X_c, X_e, logits):
        """
        Ternary Search to find n that minimizes energy.
        Assumes convexity in the range [n_min, n_max].
        """
        n_e = X_e.shape[0]
        left = self.n_min
        right = min(self.n_max, n_e)
        
        # Cache results to avoid re-computing for same n
        cache = {}
        
        def get_score(n):
            if n in cache: return cache[n]
            val = self._evaluate_n_energy(n, logits, X_t, X_c, X_e)
            cache[n] = val
            return val
        
        while right - left > 2:
            m1 = left + (right - left) // 3
            m2 = right - (right - left) // 3
            
            e1 = get_score(m1)
            e2 = get_score(m2)
            
            if e1 < e2:
                right = m2
            else:
                left = m1
                
        # Check the small remaining range
        candidates = range(left, right + 1)
        scores = [get_score(n) for n in candidates]
        best_idx = np.argmin(scores)
        
        return candidates[best_idx]

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        # 1. Estimate Optimal N
        best_n_aug = self._estimate_optimal_augmentation(rct_pool.X, ext_pool.X)
        
        # 2. Perform Final Split (Design)
        n_rct = rct_pool.X.shape[0]
        
        # Calculation: N_treat = (N_rct + N_aug) / 2
        if self.ratio_trt_before_augmentation is not None:
            n_treat = int((n_rct) * self.ratio_trt_before_augmentation)
        else:
            n_treat = int((n_rct + best_n_aug) * self.ratio_trt_after_augmentation)
        
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

class PooledEnergyMinimizer(BaseDesign):
    """
    For n = 0, ..., N_ext, finds the n that minimizes Energy distance between:
    - Pooled RCT (Treatment + Control)
    - Best matching population from External data of size n.

    Once n* optimal is found, splits RCT into treatment/control to balance effective sample sizes.
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

    def _optimize_soft_weights(self, X_pool, X_ext):
        """
        Learns soft weights for X_ext that make it match X_pool.
        Reference: EnergyAugmenter_PooledTarget.fit
        """
        X_pool_torch = torch.tensor(X_pool, dtype=torch.float32, device=self.device)
        X_e_torch = torch.tensor(X_ext, dtype=torch.float32, device=self.device)
        
        return optimize_soft_weights(
            X_source=X_e_torch,
            X_target=X_pool_torch,
            X_internal=None, # Pure matching
            target_n_aug=None,
            lr=self.lr,
            n_iter=self.n_iter
        )

    def _evaluate_n_energy(self, n, logits, X_pool, X_ext):
        """
        Samples subsets of size n based on logits and returns the best Energy Distance found.
        Reference: EnergyAugmenter_PooledTarget.sample
        """
        if n == 0: return 9999.0
        
        X_pool_torch = torch.tensor(X_pool, dtype=torch.float32, device=self.device)
        X_ext_torch = torch.tensor(X_ext, dtype=torch.float32, device=self.device)
        
        # 1. Sample Indices
        probs = F.softmax(logits, dim=0)
        probs_np = probs.cpu().numpy()
        probs_np /= probs_np.sum()
        pool_idx = np.arange(len(probs_np))
        
        batch_indices_list = [
            np.random.choice(pool_idx, size=n, replace=False, p=probs_np)
            for _ in range(self.k_best)
        ]
        
        # (k_best, n)
        batch_idx = torch.tensor(np.array(batch_indices_list), device=self.device, dtype=torch.long)
        
        # 2. Gather Candidates: (k_best, n, dim)
        X_cand_ext = X_ext_torch[batch_idx]
        
        # 3. Compute Energy
        energies = compute_batch_energy(X_cand_ext, X_pool_torch, X_internal=None)
        
        return torch.min(energies).item()

    def _search_best_n(self, X_pool, X_ext, logits):
        """
        Searches for n that minimizes the Energy distance.
        Uses Ternary Search assuming convexity of the Energy profile.
        """
        left = self.n_min
        right = min(self.n_max, X_ext.shape[0])
        
        cache = {}
        
        def get_score(n):
            if n in cache: return cache[n]
            val = self._evaluate_n_energy(n, logits, X_pool, X_ext)
            cache[n] = val
            return val
        
        # Ternary Search
        while right - left > 2:
            m1 = left + (right - left) // 3
            m2 = right - (right - left) // 3
            
            e1 = get_score(m1)
            e2 = get_score(m2)
            
            if e1 < e2:
                right = m2
            else:
                left = m1
                
        # Final fine-grained check
        candidates = range(left, right + 1)
        scores = [get_score(n) for n in candidates]
        best_idx = np.argmin(scores)
        
        return candidates[best_idx]

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        n_rct = rct_pool.X.shape[0]
        
        # 1. Optimize Weights (Gradient Descent)
        # We compare the Entire RCT (Pooled) vs External
        logits = self._optimize_soft_weights(rct_pool.X, ext_pool.X)
        
        # 2. Find Optimal N (Sampling)
        best_n_aug = self._search_best_n(rct_pool.X, ext_pool.X, logits)
        
        # 3. Perform Design Split
        # Balancing Effective Sample Size: N_treat = (N_rct + N_aug) / 2
        n_treat = int((n_rct + best_n_aug) / 2)
        n_treat = min(max(n_treat, 10), n_rct - 10) # Safety bounds
        
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