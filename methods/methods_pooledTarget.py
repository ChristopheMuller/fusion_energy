import numpy as np
import torch
import torch.nn.functional as F

class EnergyAugmenter_PooledTarget:
    def __init__(self, n_sampled=100, k_best=100, lr=0.01, n_iter=500):
        self.n_sampled = n_sampled
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.weights_ = None
        
    def fit(self, X_target, X_internal, X_external):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_t = torch.tensor(X_target, dtype=torch.float32, device=device)
        X_i = torch.tensor(X_internal, dtype=torch.float32, device=device)
        X_e = torch.tensor(X_external, dtype=torch.float32, device=device)
        
        X_pool = torch.cat([X_t, X_i], dim=0)
        
        n_e = X_e.shape[0]
        
        d_ep = torch.cdist(X_e, X_pool)
        d_ep_mean = d_ep.mean(dim=1) 
        
        d_ee = torch.cdist(X_e, X_e)

        logits = torch.zeros(n_e, requires_grad=True, device=device)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0)
            
            term1 = 2 * torch.dot(w, d_ep_mean)
            
            term2 = torch.dot(w, torch.mv(d_ee, w))
            
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
        
        X_pool = torch.cat([X_t_torch, X_i_torch], dim=0)
        X_pool_expanded = X_pool.unsqueeze(0)
        
        X_cand_ext = torch.stack([X_e_torch[idx] for idx in batch_indices])
        
        d_cp = torch.cdist(X_cand_ext, X_pool_expanded)
        term1 = 2 * d_cp.mean(dim=(1, 2))
        
        d_cc = torch.cdist(X_cand_ext, X_cand_ext)
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