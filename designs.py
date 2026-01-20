from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from structures import PotentialOutcomes, SplitData

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
                 device=None):
        self.k_folds = k_folds
        self.k_best = k_best
        self.n_min = n_min
        self.n_max = n_max
        self.lr = lr
        self.n_iter = n_iter
        self.ratio_trt_after_augmentation = ratio_trt_after_augmentation
        
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
        n_c = X_c.shape[0]
        n_t = X_t.shape[0]
        
        # Proxy n for optimization (e.g., middle of search range)
        # This helps set the 'beta' balance parameter for the loss function
        proxy_n = (self.n_min + min(self.n_max, n_e)) // 2
        total_n = n_c + proxy_n
        beta = proxy_n / total_n  # Weight of External relative to Internal

        # Precompute Distance Matrices (The Heavy Lifting)
        # We need: D_et (External-Target), D_ie (Internal-External), D_ee (External-External)
        d_et = torch.cdist(X_e, X_t)
        d_et_sum = d_et.sum(dim=1) # Sum over Target
        
        d_ie = torch.cdist(X_c, X_e)
        d_ie_sum = d_ie.sum(dim=0) # Sum over Internal
        
        d_ee = torch.cdist(X_e, X_e)

        # Optimization
        logits = torch.zeros(n_e, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0)
            
            # Energy Distance Terms
            # 1. Cross Term: (Pooled vs Target)
            # Contribution from External: w * sum(d_et)
            term1_ext = torch.dot(w, d_et_sum) / n_t
            # (Note: Internal-Target term is constant wrt w, so ignored in grad)
            
            # 2. Self Term: (Pooled vs Pooled)
            # Part A: External vs External -> w^T * D_ee * w
            term2_ee = torch.dot(w, torch.mv(d_ee, w))
            
            # Part B: Internal vs External -> w * sum(d_ie)
            term2_ie = torch.dot(w, d_ie_sum) / n_c
            
            # Weighted Energy Objective
            # Energy = 2 * E[XY] - E[XX] - E[YY]
            # We only care about terms involving w
            loss = (2 * beta * term1_ext) - (beta**2 * term2_ee + 2 * (1 - beta) * beta * term2_ie)
            
            loss.backward()
            opt.step()
            
        return logits.detach()

    def _evaluate_n_energy(self, n, logits, X_t, X_c, X_e):
        """
        Samples 'k_best' subsets of size 'n' using 'logits',
        computes the EXACT Energy Distance for the pooled set,
        returns the minimum energy found.
        """
        if n == 0: return 9999.0 # Edge case
        
        n_c = X_c.shape[0]
        n_t = X_t.shape[0]
        n_pool = n_c + n
        
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
        
        # Term 1: Pooled vs Target (Cross)
        # Pooled = [X_c (fixed), X_aug (variable)]
        # E[d(P, T)] = (Sum d(X_c, T) + Sum d(X_aug, T)) / (N_pool * N_t)
        
        # We pre-calculate d(X_c, T) sum outside if we want ultimate speed, 
        # but here we just compute on fly for clarity.
        
        # X_c_expanded: (1, n_c, dim) -> broadcast to (k_best, n_c, dim) is implicit in cdist? No.
        # Let's compute components.
        
        # A. Constant Part (Internal vs Target)
        # dist_ct_const = torch.cdist(X_c, X_t).sum() # Scalar
        
        # B. Variable Part (Augmented vs Target)
        # dist_aug_t: (k_best, n, n_t)
        X_t_expanded = X_t.unsqueeze(0) # (1, n_t, dim)
        dist_aug_t = torch.cdist(X_aug, X_t_expanded) 
        sum_aug_t = dist_aug_t.sum(dim=(1, 2)) # (k_best,)
        
        # Term 2: Pooled vs Pooled (Self)
        # E[d(P, P)] = (Sum d(C, C) + Sum d(A, A) + 2 Sum d(C, A)) / N_pool^2
        
        # A. Aug vs Aug
        dist_aa = torch.cdist(X_aug, X_aug)
        sum_aa = dist_aa.sum(dim=(1, 2)) # (k_best,)
        
        # B. Internal vs Aug
        X_c_expanded = X_c.unsqueeze(0) # (1, n_c, dim)
        dist_ca = torch.cdist(X_c_expanded, X_aug) # (k_best, n_c, n)
        sum_ca = dist_ca.sum(dim=(1, 2)) # (k_best,)
        
        # Calculate Metric (Energy Proxy)
        # We can drop constants (like d(C,C) and d(C,T)) if we only care about minimization,
        # BUT n changes, so the denominator (N_pool) changes. We must include constants.
        
        sum_ct = torch.cdist(X_c, X_t).sum()
        sum_cc = torch.cdist(X_c, X_c).sum()
        
        term_cross = (sum_ct + sum_aug_t) / (n_pool * n_t)
        term_self = (sum_cc + sum_aa + 2 * sum_ca) / (n_pool**2)
        
        # Energy = 2 * Cross - Self - (Target-Target constant)
        # We ignore Target-Target constant as it doesn't affect min
        energy = 2 * term_cross - term_self
        
        return torch.min(energy).item()

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