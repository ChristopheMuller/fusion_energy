import numpy as np
import torch
import torch.nn.functional as F
from .base import BaseAugmenter

class EnergyAugmenter_MatchingReg(BaseAugmenter):
    def __init__(self, n_sampled=100, k_best=100, lr=0.01, n_iter=500, lambda_mean=10.0):
        super().__init__()
        self.n_sampled = n_sampled
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_mean = lambda_mean  # New hyperparameter
        
    def fit(self, X_target, X_internal, X_external):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data Loading
        X_t = torch.tensor(X_target, dtype=torch.float32, device=device)
        X_i = torch.tensor(X_internal, dtype=torch.float32, device=device)
        X_e = torch.tensor(X_external, dtype=torch.float32, device=device)
        
        n_i = X_i.shape[0]
        n_t = X_t.shape[0]
        n_e = X_e.shape[0]
        
        # Scaling Factors
        total_final = n_i + self.n_sampled
        alpha = n_i / total_final
        beta = self.n_sampled / total_final

        # --- Energy Terms Pre-computation ---
        d_et = torch.cdist(X_e, X_t)
        d_et_sum = d_et.sum(dim=1) 
        d_ie = torch.cdist(X_i, X_e)
        d_ie_sum = d_ie.sum(dim=0)
        d_ee = torch.cdist(X_e, X_e)

        # --- Mean Regularization Pre-computation ---
        # Target Mean
        mu_t = X_t.mean(dim=0)
        # Internal Mean (Fixed)
        mu_i = X_i.mean(dim=0)
        
        # Optimization
        logits = torch.zeros(n_e, requires_grad=True, device=device)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0)
            
            # 1. Energy Loss Calculation (Same as before)
            term1_ext = torch.dot(w, d_et_sum) / n_t
            term1 = 2 * beta * term1_ext
            
            term2_ee = torch.dot(w, torch.mv(d_ee, w))
            term2_ie = torch.dot(w, d_ie_sum) / n_i
            term2 = (beta**2 * term2_ee + 2 * alpha * beta * term2_ie)
            
            loss_energy = term1 - term2
            
            # 2. Mean Regularization Loss
            # mu_aug = alpha * mu_i + beta * (w @ X_e)
            mu_ext_weighted = torch.mv(X_e.T, w) # Shape (d,)
            mu_aug = alpha * mu_i + beta * mu_ext_weighted
            
            # Squared Euclidean distance between means
            loss_mean = torch.sum((mu_aug - mu_t)**2)
            
            # Total Loss
            loss = loss_energy + self.lambda_mean * loss_mean
            
            loss.backward()
            opt.step()
            
        self.weights_ = F.softmax(logits, dim=0).detach().cpu().numpy()
        return self
    
    def sample(self, X_target, X_internal, X_external, Y_internal=None, Y_external=None):
        if self.weights_ is None:
            raise ValueError("Must call fit() before sample()")
        
        probs = self.weights_ / np.sum(self.weights_)
        pool_idx = np.arange(len(X_external))
        
        probs = probs / probs.sum()
        
        batch_indices = [
            np.random.choice(pool_idx, size=self.n_sampled, replace=False, p=probs)
            for _ in range(self.k_best)
        ]
        
        X_i_torch = torch.tensor(X_internal, dtype=torch.float32)
        X_e_torch = torch.tensor(X_external, dtype=torch.float32)
        X_t_torch = torch.tensor(X_target, dtype=torch.float32)
        
        X_cand_ext = torch.stack([X_e_torch[idx] for idx in batch_indices])
        
        X_i_expanded = X_i_torch.unsqueeze(0).expand(self.k_best, -1, -1)
        X_final_candidates = torch.cat([X_i_expanded, X_cand_ext], dim=1)        
        X_t_expanded = X_t_torch.unsqueeze(0)
        
        d_ct = torch.cdist(X_final_candidates, X_t_expanded)
        term1 = 2 * d_ct.mean(dim=(1, 2))
        
        d_cc = torch.cdist(X_final_candidates, X_final_candidates)
        term2 = d_cc.mean(dim=(1, 2))

        scores = term1 - term2
        best_k_idx = torch.argmin(scores).item()
        best_ext_indices = batch_indices[best_k_idx]
        
        X_aug_ext = X_external[best_ext_indices]
        X_fused = np.vstack([X_internal, X_aug_ext])
        
        # Calculate Weights
        n_i = X_internal.shape[0]
        n_e = X_external.shape[0]
        n_s = self.n_sampled
        
        # Weights vector corresponding to [Y_internal, Y_external]
        weights = np.zeros(n_i + n_e)
        
        if n_i + n_s > 0:
            w_val = 1.0 / (n_i + n_s)
            weights[:n_i] = w_val
            weights[n_i + best_ext_indices] = w_val

        if Y_internal is not None and Y_external is not None:
            Y_aug_ext = Y_external[best_ext_indices]
            Y_fused = np.concatenate([Y_internal, Y_aug_ext])
            return X_fused, Y_fused, weights
            
        return X_fused, weights