import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.ensemble import RandomForestRegressor

from structures import SplitData, EstimationResult
from metrics import compute_batch_energy
from .base import BaseEstimator

class MSEProg_MatchingEstimator(BaseEstimator):
    """
    Selects a subset of external controls that minimizes a combination of the Energy Distance 
    and a Prognostic Regularizer between the pooled control arm and the Treatment arm.
    """
    def __init__(self, k_best=100, lr=0.05, n_iter=300, lamb=1.0, device=None, n_external: int = None):
        super().__init__(n_external=n_external)
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.lamb = lamb
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        
        if n_external is not None:
            target_n = n_external
        elif self.n_external is not None:
            target_n = int(self.n_external)
        else:
            target_n = data.target_n_aug

        y1_mean = np.mean(data.Y_treat)

        # 0. Fit RF to compute prognostic scores (m)
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5,
            oob_score=True,
            n_jobs=1
        )
        rf.fit(data.X_external, data.Y_external)
        
        # Convert m(X) to tensors for fast batched computation later
        m_c = torch.as_tensor(rf.predict(data.X_control_int), dtype=torch.float32, device=self.device)
        m_e = torch.as_tensor(rf.oob_prediction_, dtype=torch.float32, device=self.device)
        m_t = torch.as_tensor(rf.predict(data.X_treat), dtype=torch.float32, device=self.device)
        
        # 1. Tensor Setup for Covariates
        X_t = torch.as_tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.as_tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.as_tensor(data.X_external, dtype=torch.float32, device=self.device)

        logits = self._optimise_soft_weights(
            X_source=X_e,
            X_target=X_t,
            X_internal=X_c,
            target_n_aug=target_n,
            m_e=m_e,
            m_t=m_t,
            m_c=m_c
        )
        probs = F.softmax(logits, dim=0).cpu().numpy()
        probs /= probs.sum()
        
        # 3. Select Best Sample (Stochastic Selection)
        # We pick the specific binary subset that minimizes energy
        best_indices, min_loss = self._select_best_sample(
            X_t, X_c, X_e, probs, target_n, m_t, m_c, m_e
        )
        
        # 4. Construct Estimation Result
        w_ext = np.zeros(n_ext)
        w_ext[best_indices] = 1.0
        w_int = probs
        
        # ATT Estimate
        y0_weighted = (np.sum(data.Y_control_int) + np.sum(data.Y_external[best_indices])) / (n_int + target_n)
        ate = y1_mean - y0_weighted
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_int,
            weights_external=w_ext,
            energy_distance=min_loss,
            estimation_time=time.time() - start_time
        )

    def _optimise_soft_weights(
        self,
        X_source: torch.Tensor,
        X_target: torch.Tensor,
        X_internal: torch.Tensor = None,
        target_n_aug: int = None,
        m_e: torch.Tensor = None,
        m_t: torch.Tensor = None,
        m_c: torch.Tensor = None,
        dist_st: torch.Tensor = None,
        dist_ss: torch.Tensor = None,
        dist_is: torch.Tensor = None
    ):
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        
        if dist_st is not None:
            d_st = dist_st
        else:
            d_st = torch.cdist(X_source, X_target)
        d_st_sum = d_st.sum(dim=1)
        
        if dist_ss is not None:
            d_ss = dist_ss
        else:
            d_ss = torch.cdist(X_source, X_source)
        
        if X_internal is not None:
            if target_n_aug is None:
                 raise ValueError()
                 
            n_internal = X_internal.shape[0]
            total_n = n_internal + target_n_aug
            
            beta = target_n_aug / total_n if total_n > 0 else 0.5
            
            if dist_is is not None:
                d_is = dist_is
            else:
                d_is = torch.cdist(X_internal, X_source)
            d_is_sum = d_is.sum(dim=0)
        else:
            beta = 1.0
            n_internal = 0
            d_is_sum = None
            
        logits = torch.zeros(n_source, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        if self.lamb > 0.0 and m_t is not None and m_e is not None:
            estim_YTreated = m_t.mean()
            sum_m_c = m_c.sum() if m_c is not None else 0.0
            n_c = m_c.shape[0] if m_c is not None else 0
            n_aug = target_n_aug if target_n_aug is not None else n_source
            total_pool_n = n_c + n_aug
        
        prev_loss = float('inf')
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0)
            
            term1 = torch.dot(w, d_st_sum) / n_target if n_target > 0 else 0.0
            term2_ss = torch.dot(w, torch.mv(d_ss, w))
            
            if X_internal is not None:
                term2_is = torch.dot(w, d_is_sum) / n_internal if n_internal > 0 else 0.0
                loss = (2 * beta * term1) - (beta**2 * term2_ss + 2 * beta * (1 - beta) * term2_is)
            else:
                loss = 2 * term1 - term2_ss
                
            if self.lamb > 0.0 and m_t is not None and m_e is not None:
                expected_m_e_sum = n_aug * torch.dot(w, m_e)
                estim_YControl = (sum_m_c + expected_m_e_sum) / total_pool_n
                regul = torch.square(estim_YTreated - estim_YControl)
                loss = loss + self.lamb * regul
                
            loss.backward()
            opt.step()
            
            if abs(prev_loss - loss.item()) < 1e-6:
                break
            prev_loss = loss.item()
            
        return logits.detach()

    def _select_best_sample(self, X_t, X_c, X_e, probs, n_sampled, m_t, m_c, m_e):
        pool_idx = np.arange(len(probs))
        
        batch_indices = [
            np.random.choice(pool_idx, size=n_sampled, replace=False, p=probs)
            for _ in range(self.k_best)
        ]
        batch_idx_tensor = torch.as_tensor(np.array(batch_indices), device=self.device, dtype=torch.long)
        
        # B. Prepare Batches
        X_source_batch = X_e[batch_idx_tensor] # (k, n, dim)
        
        # C. Compute Energy (Metrics API)
        energies = compute_batch_energy(
            X_source_batch=X_source_batch,
            X_target=X_t,
            X_internal=X_c
        )
        
        m_e_batch = m_e[batch_idx_tensor] 
        
        estim_YTreated = m_t.mean()
        estim_YControl = (m_c.sum() + m_e_batch.sum(dim=1)) / (len(m_c) + n_sampled)
        
        regul = torch.square(estim_YTreated - estim_YControl)
        total_loss = energies + self.lamb * regul
        
        best_k_idx = torch.argmin(total_loss).item()
        min_loss = total_loss[best_k_idx].item()
        
        return batch_indices[best_k_idx], min_loss