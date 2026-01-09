import numpy as np
import torch
import torch.nn.functional as F

class EnergyAugmenter:
    def __init__(self, n_augment=100, k_best=100, lr=0.01, n_iter=500):
        self.n_augment = n_augment
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
        
        n_i = X_i.shape[0]
        n_e = X_e.shape[0]
        n_t = X_t.shape[0]
        
        # --- FIX 1: Correct Mass Ratios ---
        # The mixture weights must reflect the FINAL COHORT composition
        total_final_size = n_i + self.n_augment
        alpha = n_i / total_final_size              # Mass of Internal
        beta = self.n_augment / total_final_size    # Mass of External
        
        # --- FIX 2: Precompute Distance Matrices (Speedup) ---
        # 1. Internal terms (Constant)
        d_it_mean = torch.cdist(X_i, X_t).mean()
        d_ii_mean = torch.cdist(X_i, X_i).mean()
        
        # 2. Cross interaction terms (Fixed matrices)
        d_et = torch.cdist(X_e, X_t)  # Shape (n_e, n_t)
        d_ee = torch.cdist(X_e, X_e)  # Shape (n_e, n_e)
        d_ie = torch.cdist(X_i, X_e)  # Shape (n_i, n_e)
        
        # Optimizer
        logits = torch.zeros(n_e, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0) # Shape (n_e,)
            
            # --- TERM 1: Cross Energy (Union vs Target) ---
            # E[||U - T||] = alpha * E[||I - T||] + beta * E[||E_w - T||]
            
            # w.unsqueeze(1) * d_et broadcasts w to rows of d_et
            term1_ext = torch.sum(w.unsqueeze(1) * d_et) / n_t
            term1 = 2 * (alpha * d_it_mean + beta * term1_ext)
            
            # --- TERM 2: Self Energy (Union vs Union) ---
            # (alpha*I + beta*E)^2 = alpha^2*II + beta^2*EE + 2*alpha*beta*IE
            
            # Weighted Energy of External
            term2_ee = torch.sum(w.unsqueeze(1) * d_ee * w.unsqueeze(0))
            
            # Interaction Internal-External
            # Mean over I (rows), Weighted Sum over E (cols)
            # sum(d_ie * w) -> sums everything. Divide by n_i to get expectation over I.
            term2_ie = torch.sum(d_ie * w.unsqueeze(0)) / n_i
            
            term2 = (alpha**2 * d_ii_mean + 
                     beta**2 * term2_ee + 
                     2 * alpha * beta * term2_ie)
            
            loss = term1 - term2
            loss.backward()
            opt.step()
            
        self.weights_ = F.softmax(logits, dim=0).detach().numpy()
        return self

    def sample(self, X_target, X_internal, X_external, Y_external=None):
        # Best-of-K Selection
        best_dist = np.inf
        best_idx = None
        
        # Base probabilities from optimization
        probs = self.weights_ / np.sum(self.weights_)
        pool_idx = np.arange(len(X_external))
        
        X_t_torch = torch.tensor(X_target, dtype=torch.float32)
        X_i_torch = torch.tensor(X_internal, dtype=torch.float32)
        X_e_torch = torch.tensor(X_external, dtype=torch.float32)
        
        for k in range(self.k_best):
            # Sample WITHOUT replacement
            chosen_idx = np.random.choice(
                pool_idx, size=self.n_augment, replace=False, p=probs
            )
            
            # Form the cohort: Internal + Selected External
            X_cand = torch.cat([X_i_torch, X_e_torch[chosen_idx]])
            
            # Compute Metric (Energy Distance to Target)
            # Unweighted, because this is the final analysis cohort
            dist = self._energy_dist_torch(X_cand, X_t_torch).item()
            
            if dist < best_dist:
                best_dist = dist
                best_idx = chosen_idx
        
        if Y_external is not None:
            return X_external[best_idx], Y_external[best_idx]
        return X_external[best_idx]