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

    def sample(self, X_target, X_internal, X_external, Y_internal=None, Y_external=None, strategy='weighted', n_sampled=None):
        if self.weights_ is None:
            raise ValueError("Must call fit() before sample()")
            
        # Use instance n_sampled if not provided
        if n_sampled is None:
            n_sampled = self.n_sampled
            
        # Base probabilities from optimization
        probs = self.weights_ / np.sum(self.weights_)
        
        # Pool Data for indexing
        # X_pool is used to extract the chosen rows
        X_pool = np.vstack([X_internal, X_external])
        
        n_pool = X_pool.shape[0]
        pool_idx = np.arange(n_pool)
        
        # Ensure we don't sample more than available
        if n_sampled > n_pool:
             n_sampled = n_pool

        if strategy in ['weighted', 'top_n']:
            if strategy == 'weighted':
                # Sample WITHOUT replacement from the WHOLE pool
                chosen_idx = np.random.choice(
                    pool_idx, size=n_sampled, replace=False, p=probs
                )
            elif strategy == 'top_n':
                # Select the indices with the largest weights
                sorted_idx = np.argsort(probs)
                chosen_idx = sorted_idx[-n_sampled:]
            
            # Form the cohort
            X_chosen = X_pool[chosen_idx]
            if Y_internal is not None and Y_external is not None:
                Y_pool = np.concatenate([Y_internal, Y_external])
                Y_chosen = Y_pool[chosen_idx]
                return X_chosen, Y_chosen
            return X_chosen

        elif strategy in ['weighted_hybrid', 'top_n_hybrid']:
            n_internal = X_internal.shape[0]
            
            # Split weights
            w_int = self.weights_[:n_internal]
            w_ext = self.weights_[n_internal:]
            
            # Indices relative to their own arrays
            idx_int = np.arange(n_internal)
            n_external = X_external.shape[0]
            idx_ext = np.arange(n_external)

            # Case 1: n_sampled <= n_internal (Sample only from Internal)
            if n_sampled <= n_internal:
                # Renormalize internal weights
                if w_int.sum() > 0:
                    p_int = w_int / w_int.sum()
                else:
                    p_int = np.ones(n_internal) / n_internal
                
                if strategy == 'weighted_hybrid':
                    chosen_int_sub_idx = np.random.choice(idx_int, size=n_sampled, replace=False, p=p_int)
                else: # top_n_hybrid
                    sorted_sub_idx = np.argsort(p_int)
                    chosen_int_sub_idx = sorted_sub_idx[-n_sampled:]
                
                X_chosen = X_internal[chosen_int_sub_idx]
                if Y_internal is not None:
                    Y_chosen = Y_internal[chosen_int_sub_idx]
                    return X_chosen, Y_chosen
                return X_chosen

            # Case 2: n_sampled > n_internal (All Internal + Sample from External)
            else:
                n_ext_needed = n_sampled - n_internal
                
                # Cap if requested more than available external
                if n_ext_needed > n_external:
                    n_ext_needed = n_external
                
                # Renormalize external weights
                if w_ext.sum() > 0:
                    p_ext = w_ext / w_ext.sum()
                else:
                    p_ext = np.ones(n_external) / n_external
                
                if strategy == 'weighted_hybrid':
                    chosen_ext_sub_idx = np.random.choice(idx_ext, size=n_ext_needed, replace=False, p=p_ext)
                else: # top_n_hybrid
                    sorted_sub_idx = np.argsort(p_ext)
                    chosen_ext_sub_idx = sorted_sub_idx[-n_ext_needed:]
                
                # Combine: All Internal + Chosen External
                X_chosen = np.vstack([X_internal, X_external[chosen_ext_sub_idx]])
                
                if Y_internal is not None and Y_external is not None:
                    Y_chosen = np.concatenate([Y_internal, Y_external[chosen_ext_sub_idx]])
                    return X_chosen, Y_chosen
                
                return X_chosen

        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'weighted', 'top_n', 'weighted_hybrid', or 'top_n_hybrid'.")