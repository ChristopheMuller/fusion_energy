import numpy as np
import torch
import torch.nn.functional as F

from structures import SplitData, EstimationResult
from metrics import optimise_soft_weights
from .base import BaseEstimator

class EnergyPooledWeightingEstimator(BaseEstimator):

    def __init__(self, lr=0.05, n_iter=300, device=None):
        super().__init__()
        self.lr = lr
        self.n_iter = n_iter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def estimate(self, data: SplitData, **kwargs) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        n_trt = data.X_treat.shape[0]
        n_rct = n_int + n_trt
        
        y1_mean = np.mean(data.Y_treat)
        
        # 1. Tensor Setup
        X_t_t = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c_t = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e_t = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)

        X_rct_t = torch.cat([X_t_t, X_c_t], dim=0)

        logits_ext = torch.zeros(n_ext, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([logits_ext], lr=self.lr)

        # distances
        d_er = torch.cdist(X_e_t, X_rct_t)
        d_tt = torch.cdist(X_t_t, X_t_t)
        d_ee = torch.cdist(X_e_t, X_e_t)
        sum_er_per_ext = d_er.sum(dim=1)

        prev_loss = float('inf')

        for it in range(self.n_iter):

            opt.zero_grad()

            w = torch.softmax(logits_ext, dim=0)

            # 2 * E[d(Ext_w, RCT)] - E[d(Ext_w, Ext_w)]
            term_cross = 2 * torch.dot(w, sum_er_per_ext) / n_rct
            term_self = torch.dot(w , torch.mv(d_ee, w))

            loss = term_cross - term_self

            loss.backward()
            opt.step()

            if abs(prev_loss - loss.item()) < 1e-6:
                break
            prev_loss = loss.item()


        w_ext = F.softmax(logits_ext, dim=0).detach().cpu().numpy()

        y0_ext = data.Y_external
        y0_weighted_mean = np.sum(w_ext * y0_ext) / np.sum(w_ext)
        ate = y1_mean - y0_weighted_mean
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_ext,
            weights_external=w_ext/ np.sum(w_ext),
            energy_distance=None
        )