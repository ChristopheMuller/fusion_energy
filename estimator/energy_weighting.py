import numpy as np
import torch
import torch.nn.functional as F
import time
from structures import SplitData, EstimationResult
from .base import BaseEstimator

class Energy_WeightingEstimator(BaseEstimator):
    def __init__(self, lr=0.05, n_iter=300, device=None):
        super().__init__()
        self.lr = lr
        self.n_iter = n_iter
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
        n_int = data.X_control_int.shape[0]
        n_t = data.X_treat.shape[0]
        
        y1_mean = np.mean(data.Y_treat)
        
        X_t = torch.as_tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_pool = torch.as_tensor(np.concatenate([data.X_control_int, data.X_external], axis=0),
                              dtype=torch.float32, device=self.device)
        Y_pool = np.concatenate([data.Y_control_int, data.Y_external], axis=0)

        n_pool = X_pool.shape[0]
        logits = torch.zeros(n_pool, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([logits], lr=self.lr)

        d_pt = torch.cdist(X_pool, X_t)
        sum_pt_per_unit = d_pt.sum(dim=1)
        d_pp = torch.cdist(X_pool, X_pool)

        prev_loss = float('inf')
        for _ in range(self.n_iter):
            opt.zero_grad()
            
            w = F.softmax(logits, dim=0)
            
            term_cross = torch.dot(w, sum_pt_per_unit) / n_t
            term_self = torch.dot(w, torch.mv(d_pp, w))
            
            loss = 2 * term_cross - term_self
            
            loss.backward()
            opt.step()
            
            if abs(prev_loss - loss.item()) < 1e-6:
                break
            prev_loss = loss.item()

        final_w = F.softmax(logits, dim=0).detach().cpu().numpy()
        
        y0_weighted_mean = np.average(Y_pool, weights=final_w)
        ate = y1_mean - y0_weighted_mean
        
        w_int = final_w[:n_int]
        w_ext = final_w[n_int:]
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_ext,
            weights_external=w_ext * n_pool,
            energy_distance=prev_loss,
            estimation_time=time.time() - start_time
        )
