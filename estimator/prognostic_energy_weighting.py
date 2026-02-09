import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from structures import SplitData, EstimationResult
from .base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor


class PrognosticEnergyWeightingEstimator(BaseEstimator):
    """
    Estimates ATE by weighting the pooled control arm (internal + external).
    
    Weights are optimized to minimize:
    1. Energy Distance between treatment and weighted pooled control (covariate balance)
    2. Energy Distance between RCT control and weighted external prognostic scores (outcome regularization)
        
    Loss = ED_covariates(Treat, Pooled_Control^w) + lambda * ED_prognosis(RCT_Control, External^w)
    """
    
    def __init__(self, lamb=0.1, lr_weights=0.01, lr_prog=0.01, 
                 n_iter_weights=600, n_iter_prog=100,
                 prog_hidden_dim=64, device=None, verbose=False):
        super().__init__()
        self.lamb = lamb
        self.lr_weights = lr_weights
        self.lr_prog = lr_prog
        self.n_iter_weights = n_iter_weights
        self.n_iter_prog = n_iter_prog
        self.prog_hidden_dim = self._check_hidden_dim(prog_hidden_dim)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def _check_hidden_dim(self, dim):
        return int(dim)

    def _energy_distance_precomputed(self, d_XY, d_XX, d_YY, w1, w2):
        term1 = 2 * torch.sum(w1.unsqueeze(1) * w2.unsqueeze(0) * d_XY)
        term2 = torch.sum(w1.unsqueeze(1) * w1.unsqueeze(0) * d_XX)
        term3 = torch.sum(w2.unsqueeze(1) * w2.unsqueeze(0) * d_YY)
        
        return term1 - term2 - term3
    
    def _train_prognostic_model(self, X_ext, Y_ext):
        X_np = X_ext.detach().cpu().numpy()
        Y_np = Y_ext.detach().cpu().numpy().ravel()
        
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            n_jobs=-1,
        )
        rf.fit(X_np, Y_np)
        
        class RFWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
                
            def __call__(self, x):
                x_np = x.detach().cpu().numpy()
                preds = self.model.predict(x_np)
                return torch.tensor(preds, dtype=torch.float32, device=self.device).view(-1, 1)
                
        return RFWrapper(rf, self.device)
    
    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
        n_int = data.X_control_int.shape[0]
        
        X_treat = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_control_int = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_external = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)
        Y_external = torch.tensor(data.Y_external, dtype=torch.float32, device=self.device).view(-1, 1)
        
        X_pooled_control = torch.cat([X_control_int, X_external], dim=0)
        Y_pooled_control = np.concatenate([data.Y_control_int, data.Y_external], axis=0)
        
        prog_model = self._train_prognostic_model(X_external, Y_external)
        
        with torch.no_grad():
            m_control_int = prog_model(X_control_int)
            m_external = prog_model(X_external)
        
        d_treat_pooled = torch.cdist(X_treat, X_pooled_control, p=2)
        d_pooled_pooled = torch.cdist(X_pooled_control, X_pooled_control, p=2)
        d_treat_treat = torch.cdist(X_treat, X_treat, p=2)
        
        d_control_external = torch.cdist(m_control_int, m_external, p=2)
        d_control_control = torch.cdist(m_control_int, m_control_int, p=2)
        d_external_external = torch.cdist(m_external, m_external, p=2)
        
        n_treat = X_treat.shape[0]
        w_treat_uniform = torch.ones(n_treat, device=self.device) / n_treat
        w_control_uniform = torch.ones(m_control_int.shape[0], device=self.device) / m_control_int.shape[0]
        
        n_pool = X_pooled_control.shape[0]
        logits = torch.zeros(n_pool, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.AdamW([logits], lr=self.lr_weights, weight_decay=1e-4)
        # OneCycleLR is great for rapid convergence in fixed-data optimization
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr_weights * 5, 
            total_steps=self.n_iter_weights, pct_start=0.1
        )
        
        if self.verbose:
            print(f"\n--- Optimizing Weights (AdamW + OneCycle) ---")

        prev_loss = float('inf')

        with torch.no_grad():
            original_cov_loss = self._energy_distance_precomputed(
                d_treat_pooled, d_treat_treat, d_pooled_pooled,
                w_treat_uniform, torch.ones(n_pool, device=self.device) / n_pool
            ).item()
            
            # Using w_control_uniform for both ensures we see the 'uniform' imbalance
            original_prog_loss = self._energy_distance_precomputed(
                d_control_external, d_control_control, d_external_external,
                w_control_uniform, torch.ones(m_external.shape[0], device=self.device) / m_external.shape[0]
            ).item()
        
        # Use a slightly safer epsilon or max(val, 1e-10)
        s_cov = max(original_cov_loss, 1e-10)
        s_prog = max(original_prog_loss, 1e-10)
        
        for epoch in range(self.n_iter_weights):
            optimizer.zero_grad()

            w = F.softmax(logits, dim=0)
            w_ext = w[n_int:]
            
            # Normalize external weights so the prognostic ED doesn't collapse to zero
            w_ext_norm = w_ext / (torch.sum(w_ext) + 1e-10)
            
            loss_cov = self._energy_distance_precomputed(
                d_treat_pooled, d_treat_treat, d_pooled_pooled,
                w_treat_uniform, w
            )
            
            loss_prog = self._energy_distance_precomputed(
                d_control_external, d_control_control, d_external_external,
                w_control_uniform, w_ext_norm
            )

            loss = loss_cov / s_cov + self.lamb * loss_prog / s_prog

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if self.verbose and (epoch % 100 == 0 or epoch == self.n_iter_weights - 1):
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | Total: {loss.item():.6f} | Cov: {loss_cov.item()/s_cov:.6f} | Prog: {loss_prog.item()/s_prog:.6f} | LR: {curr_lr:.5f}")

            if abs(prev_loss - loss.item()) < 1e-8 and epoch > self.n_iter_weights // 4:
                if self.verbose:
                    print(f"Converged at epoch {epoch} with loss {loss.item():.6f}")
                break
            prev_loss = loss.item()
        
        final_w = F.softmax(logits, dim=0).detach().cpu().numpy()
        y1_mean = np.mean(data.Y_treat)
        y0_weighted_mean = np.average(Y_pooled_control, weights=final_w)
        ate = y1_mean - y0_weighted_mean
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=final_w[n_int:],
            weights_external=final_w[n_int:] * n_pool,
            energy_distance=loss_cov.item(),
            estimation_time=time.time() - start_time
        )