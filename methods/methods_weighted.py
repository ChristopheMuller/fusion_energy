import numpy as np
import torch
import torch.nn.functional as F

class EnergyAugmenter_Weighted:
    def __init__(self, n_sampled=100, k_best=100, lr=0.01, n_iter=500):
        self.n_sampled = n_sampled
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.weights_ = None
        
    def fit(self, X_target, X_internal, X_external):
        # 1. Move to tensor (ensure device is consistent)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.tensor(X_target, dtype=torch.float32, device=device)
        X_i = torch.tensor(X_internal, dtype=torch.float32, device=device)
        X_e = torch.tensor(X_external, dtype=torch.float32, device=device)
        
        n_i = X_i.shape[0]
        n_e = X_e.shape[0]
        n_t = X_t.shape[0]
        
        total_final = n_i + self.n_sampled
        # alpha = n_i / total_final  <-- Constant scaling factors
        beta = self.n_sampled / total_final

        # 2. Pre-compute Constant Summaries (The "Heavy Lifting")
        # Instead of summing inside the loop, we sum ONCE here.
        
        # Term 1 Part: External vs Target
        # Original: sum(w * d_et) 
        # Optimized: w * sum(d_et, dim=1)
        d_et = torch.cdist(X_e, X_t)
        d_et_sum = d_et.sum(dim=1)  # Shape (N_e,)

        # Term 2 Part: Internal vs External
        # Original: sum(d_ie * w)
        # Optimized: w * sum(d_ie, dim=0)
        d_ie = torch.cdist(X_i, X_e)
        d_ie_sum = d_ie.sum(dim=0)  # Shape (N_e,)
        
        # Term 2 Part: External vs External (Quadratic)
        # We need the full matrix for w^T * D * w, but we compute it once.
        d_ee = torch.cdist(X_e, X_e)

        # 3. Optimization Loop (Now strictly O(N_e^2) or O(N_e))
        logits = torch.zeros(n_e, requires_grad=True, device=device)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0)
            
            # --- Term 1: Cross Energy ---
            # We removed the constant (alpha * d_it_mean) as it has 0 gradient
            # Dot product is much faster than matrix broadcasting
            term1_ext = torch.dot(w, d_et_sum) / n_t
            term1 = 2 * beta * term1_ext
            
            # --- Term 2: Self Energy ---
            # We removed the constant (alpha^2 * d_ii_mean)
            
            # External self-interaction: w^T * D_ee * w
            # torch.mv(matrix, vector) is faster than broadcasting
            term2_ee = torch.dot(w, torch.mv(d_ee, w))
            
            # Internal-External interaction: Dot product
            term2_ie = torch.dot(w, d_ie_sum) / n_i
            
            term2 = (beta**2 * term2_ee + 
                     2 * (1 - beta) * beta * term2_ie) # Note: alpha = 1 - beta
            
            loss = term1 - term2
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
        
        if Y_internal is not None and Y_external is not None:
            Y_aug_ext = Y_external[best_ext_indices]
            Y_fused = np.concatenate([Y_internal, Y_aug_ext])
            return X_fused, Y_fused
            
        return X_fused