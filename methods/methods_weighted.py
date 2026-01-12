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
        X_t = torch.tensor(X_target, dtype=torch.float32)
        X_i = torch.tensor(X_internal, dtype=torch.float32)
        X_e = torch.tensor(X_external, dtype=torch.float32)
        
        n_i = X_i.shape[0]
        n_e = X_e.shape[0]
        n_t = X_t.shape[0]
        
        total_final = n_i + self.n_sampled
        alpha = n_i / total_final
        beta = self.n_sampled / total_final

        # 1. Internal vs Target (Constant bias)
        d_it_mean = torch.cdist(X_i, X_t).mean()
        
        # 2. Internal vs Internal (Constant variance)
        d_ii_mean = torch.cdist(X_i, X_i).mean()
        
        # 3. External matrices
        d_et = torch.cdist(X_e, X_t)
        d_ee = torch.cdist(X_e, X_e)
        d_ie = torch.cdist(X_i, X_e)

        logits = torch.zeros(n_e, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0) # Weights for External ONLY
            
            # --- Term 1: Cross Energy (Union vs Target) ---
            # E[U-T] = alpha * E[I-T] + beta * E[E_w - T]
            term1_ext = torch.sum(w.unsqueeze(1) * d_et) / n_t
            term1 = 2 * (alpha * d_it_mean + beta * term1_ext)
            
            # --- Term 2: Self Energy (Union vs Union) ---
            # E[U-U] = alpha^2 E[I-I] + beta^2 E[E_w - E_w] + 2 alpha beta E[I - E_w]
            
            # External self-interaction
            term2_ee = torch.sum(w.unsqueeze(1) * d_ee * w.unsqueeze(0))
            
            # Interaction Internal-External (Key for balancing!)
            # We average over Internal (fixed), sum weighted External
            term2_ie = torch.sum(d_ie * w.unsqueeze(0)) / n_i
            
            term2 = (alpha**2 * d_ii_mean + 
                     beta**2 * term2_ee + 
                     2 * alpha * beta * term2_ie)
            
            loss = term1 - term2
            loss.backward()
            opt.step()
            
        self.weights_ = F.softmax(logits, dim=0).detach().numpy()
        
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