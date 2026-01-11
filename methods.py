import numpy as np
import torch
import torch.nn.functional as F

class EnergyAugmenter:
    def __init__(self, n_sampled=100, k_best=100, lr=0.01, n_iter=500):
        self.n_sampled = n_sampled
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.weights_ = None
        
    def _energy_dist_torch(self, X_sample, X_target, weights=None):
        # 2 * E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        # Y is fixed, so we ignore E[||Y-Y'||]
        
        n_s = X_sample.shape[0]
        n_t = X_target.shape[0]
        
        if weights is None:
            weights = torch.ones(n_s, device=X_sample.device) / n_s
            
        # Cross term: 2 * w * ||X - Y||
        d_xy = torch.cdist(X_sample, X_target, p=2)
        term1 = 2 * torch.sum(weights.unsqueeze(1) * d_xy) / n_t
        
        # Self term: w * ||X - X|| * w
        d_xx = torch.cdist(X_sample, X_sample, p=2)
        term2 = torch.sum(weights.unsqueeze(1) * d_xx * weights.unsqueeze(0))
        
        return term1 - term2

    def fit(self, X_target, X_internal, X_external):
        # Move data to device once
        X_t = torch.tensor(X_target, dtype=torch.float32)
        X_i = torch.tensor(X_internal, dtype=torch.float32)
        X_e = torch.tensor(X_external, dtype=torch.float32)
        
        # Pool the source data (Internal + External)
        X_pool = torch.cat([X_i, X_e], dim=0)
        n_pool = X_pool.shape[0]
        n_t = X_t.shape[0]

        # Precompute distance matrices
        # Pool vs Target
        d_pt = torch.cdist(X_pool, X_t) # (n_pool, n_t)
        
        # Pool vs Pool
        d_pp = torch.cdist(X_pool, X_pool) # (n_pool, n_pool)
        
        # Optimizer
        logits = torch.zeros(n_pool, requires_grad=True)

        with torch.no_grad():
            logits[:X_i.shape[0]] = 1.0
            logits[X_i.shape[0]:] = 0.0
        
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0) # Shape (n_pool,)
            
            # Term 1: Cross Energy (Pool vs Target)
            # 2 * E[||P - T||] = 2 * sum(w_i * d(p_i, t_j)) / n_t
            term1 = 2 * torch.sum(w.unsqueeze(1) * d_pt) / n_t
            
            # Term 2: Self Energy (Pool vs Pool)
            # E[||P - P'||] = sum(w_i * w_j * d(p_i, p_j))
            term2 = torch.sum(w.unsqueeze(1) * d_pp * w.unsqueeze(0))
            
            loss = term1 - term2
            loss.backward()
            opt.step()
            
        self.weights_ = F.softmax(logits, dim=0).detach().numpy()
        return self

    def sample(self, X_target, X_internal, X_external, Y_internal=None, Y_external=None):
        # Best-of-K Selection
        best_dist = np.inf
        best_idx = None
        
        # Base probabilities from optimization
        probs = self.weights_ / np.sum(self.weights_)
        
        # Pool Data
        X_i_torch = torch.tensor(X_internal, dtype=torch.float32)
        X_e_torch = torch.tensor(X_external, dtype=torch.float32)
        X_pool_torch = torch.cat([X_i_torch, X_e_torch], dim=0)
        
        X_t_torch = torch.tensor(X_target, dtype=torch.float32)
        
        n_i = X_internal.shape[0]
        n_pool = X_pool_torch.shape[0]
        pool_idx = np.arange(n_pool)
        
        # Determine total size to sample
        total_size = self.n_sampled
        
        # Ensure we don't sample more than available
        if total_size > n_pool:
             total_size = n_pool

        for k in range(self.k_best):
            # Sample WITHOUT replacement from the WHOLE pool
            chosen_idx = np.random.choice(
                pool_idx, size=total_size, replace=False, p=probs
            )
            
            # Form the cohort
            X_cand = X_pool_torch[chosen_idx]
            
            # Compute Metric (Energy Distance to Target)
            dist = self._energy_dist_torch(X_cand, X_t_torch).item()
            
            if dist < best_dist:
                best_dist = dist
                best_idx = chosen_idx
        
        # Handle returns
        # We need to construct the return arrays from pooled internal/external
        X_pool = np.vstack([X_internal, X_external])
        X_chosen = X_pool[best_idx]
        
        if Y_internal is not None and Y_external is not None:
            Y_pool = np.concatenate([Y_internal, Y_external])
            Y_chosen = Y_pool[best_idx]
            return X_chosen, Y_chosen
            
        return X_chosen