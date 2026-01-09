import numpy as np
import torch
import torch.nn.functional as F

class EnergyAugmenter:
    def __init__(self, n_augment, k_best=100, lr=0.01, n_iter=500):
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
        X_t = torch.tensor(X_target, dtype=torch.float32)
        X_i = torch.tensor(X_internal, dtype=torch.float32)
        X_e = torch.tensor(X_external, dtype=torch.float32)
        
        n_i = X_i.shape[0]
        n_e = X_e.shape[0]
        total_size = n_i + n_e
        
        # We only optimize weights for External
        # Internal units have fixed weight proportional to their count
        # or fixed to 1/total_size per unit. 
        # Strategy: Optimize mixing weights for External to minimize Energy of Union.
        
        logits = torch.zeros(n_e, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        # Fixed part of the distribution (Internal)
        # We treat the 'sample' as a concatenation of Internal and Weighted External
        
        target_mass_int = n_i / total_size
        target_mass_ext = n_e / total_size
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            
            w_ext_raw = F.softmax(logits, dim=0)
            
            # Combine weights: Internal (uniform) + External (learned)
            # Rescale so they sum to 1 over the full set
            
            # Note: This is an approximation. We want the UNION to look like Target.
            # Ideally, we sample batch from Ext using w_ext, add all Int, compute distance.
            # Differentiable approximation: weighted average of distances.
            
            # Calculate Cross Term (Union vs Target)
            # E[||U - T||] = (n_i * E[||I - T||] + n_e * E[||E_w - T||]) / (n_i + n_e)
            
            d_it = torch.cdist(X_i, X_t).mean()
            
            d_et = torch.cdist(X_e, X_t)
            term1_ext = torch.sum(w_ext_raw.unsqueeze(1) * d_et) / X_t.shape[0]
            
            term1 = 2 * (target_mass_int * d_it + target_mass_ext * term1_ext)
            
            # Calculate Self Term (Union vs Union)
            # Decomposes into I-I, E-E, and I-E interactions
            
            d_ii = torch.cdist(X_i, X_i).mean()
            
            d_ee = torch.cdist(X_e, X_e)
            term2_ee = torch.sum(w_ext_raw.unsqueeze(1) * d_ee * w_ext_raw.unsqueeze(0))
            
            d_ie = torch.cdist(X_i, X_e)
            term2_ie = torch.sum(d_ie * w_ext_raw.unsqueeze(0)) / n_i # Mean over I, weighted sum over E
            
            term2 = (target_mass_int**2 * d_ii + 
                     target_mass_ext**2 * term2_ee + 
                     2 * target_mass_int * target_mass_ext * term2_ie)
            
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