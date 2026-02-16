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
    2. Energy Distance between treatment and weighted pooled control prognostic scores (outcome regularization)
        
    Loss = ED_covariates(Treat, Pooled_Control^w) + lambda * ED_prognosis(Treat, Pooled_Control^w)
    """
    
    def __init__(self, lamb=0.1, lr_weights=0.01, n_iter_weights=600, 
                 device=None, verbose=False):
        super().__init__()
        self.lamb = lamb
        self.lr_weights = lr_weights
        self.n_iter_weights = n_iter_weights
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def _train_prognostic_model(self, X_ext, Y_ext):
        X_np = X_ext.detach().cpu().numpy()
        Y_np = Y_ext.detach().cpu().numpy().ravel()
        
        rf = RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=-1)
        rf.fit(X_np, Y_np)
        
        class RFWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
                
            def __call__(self, x):
                x_np = x.detach().cpu().numpy()
                preds = self.model.predict(x_np)
                return torch.as_tensor(preds, dtype=torch.float32, device=self.device).view(-1, 1)
        
        return RFWrapper(rf, self.device)
    
    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
        
        # --- 1. Data Prep ---
        X_treat = torch.as_tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_control_int = torch.as_tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_external = torch.as_tensor(data.X_external, dtype=torch.float32, device=self.device)
        Y_external = torch.as_tensor(data.Y_external, dtype=torch.float32, device=self.device).view(-1, 1)
        
        n_treat = X_treat.shape[0]
        n_int = X_control_int.shape[0]
        n_ext = X_external.shape[0]
        n_pool = n_int + n_ext
        
        X_pooled_control = torch.cat([X_control_int, X_external], dim=0)
        Y_pooled_control = np.concatenate([data.Y_control_int, data.Y_external], axis=0)


        # --- 2. Prognostic Model ---
        prog_model = self._train_prognostic_model(X_external, Y_external)
        with torch.no_grad():
            m_pooled = prog_model(X_pooled_control) # (n_pool, 1)
            m_treat = prog_model(X_treat)           # (n_treat, 1) - TARGET

        # --- 3. Precompute Distance Matrices ---
        with torch.no_grad():
            d_XX_pool_treat = torch.cdist(X_pooled_control, X_treat, p=2) 
            d_XX_pool_pool  = torch.cdist(X_pooled_control, X_pooled_control, p=2)
            
            term1_linear_X = (2.0 / n_treat) * d_XX_pool_treat.sum(dim=1) 
            
            d_mm_pool_treat = torch.cdist(m_pooled, m_treat, p=2)
            d_mm_pool_pool  = torch.cdist(m_pooled, m_pooled, p=2)
            
            term1_linear_m = (2.0 / n_treat) * d_mm_pool_treat.sum(dim=1)
            
            w_unif = torch.ones(n_pool, device=self.device) / n_pool
            
            base_cov = (torch.dot(w_unif, term1_linear_X) - torch.dot(w_unif, torch.mv(d_XX_pool_pool, w_unif))).item()
            base_prog = (torch.dot(w_unif, term1_linear_m) - torch.dot(w_unif, torch.mv(d_mm_pool_pool, w_unif))).item()
            
            scale_cov = max(abs(base_cov), 1e-6)
            scale_prog = max(abs(base_prog), 1e-6)

        # --- 4. Optimization Loop ---
        logits = torch.zeros(n_pool, requires_grad=True, device=self.device)
        optimizer = torch.optim.AdamW([logits], lr=self.lr_weights, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr_weights * 5, 
            total_steps=self.n_iter_weights, pct_start=0.1
        )
        
        prev_loss = float('inf')
        
        if self.verbose:
            print(f"\n--- Optimizing Weights (Dual Objective) ---")

        for epoch in range(self.n_iter_weights):
            optimizer.zero_grad()
            w = F.softmax(logits, dim=0)
            
            # 1. Covariate Energy Distance
            # 2 * E[d(X_pool, X_treat)] - E[d(X_pool, X_pool)]
            loss_cov_linear = torch.dot(w, term1_linear_X)
            loss_cov_quadratic = torch.dot(w, torch.mv(d_XX_pool_pool, w))
            loss_cov = loss_cov_linear - loss_cov_quadratic

            # 2. Prognostic Energy Distance
            # 2 * E[d(m_pool, m_treat)] - E[d(m_pool, m_pool)]
            loss_prog_linear = torch.dot(w, term1_linear_m)
            loss_prog_quadratic = torch.dot(w, torch.mv(d_mm_pool_pool, w))
            loss_prog = loss_prog_linear - loss_prog_quadratic
            
            # Combined Loss
            total_loss = (loss_cov / scale_cov) + self.lamb * (loss_prog / scale_prog)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            new_loss = total_loss.item()
            if epoch > 10 and abs(prev_loss - new_loss) < 1e-6:
                if self.verbose: 
                    print(f"Converged at epoch {epoch:4d} | Total: {new_loss:.6f} | " f"Cov: {loss_cov.item():.6f} | Prog: {loss_prog.item():.6f}")
                break
            
            if self.verbose and (epoch % 100 == 0):
                print(f"Epoch {epoch:4d} | Total: {new_loss:.6f} | "
                      f"Cov: {loss_cov.item():.6f} | Prog: {loss_prog.item():.6f}")

        # --- 5. Result Extraction ---
        final_w = F.softmax(logits, dim=0).detach().cpu().numpy()
        
        # ATE Calculation using weighted pooled control
        y1_mean = np.mean(data.Y_treat)
        y0_weighted_mean = np.average(Y_pooled_control, weights=final_w)
        ate = y1_mean - y0_weighted_mean
        
        # Extract weights for external units only (for analysis/reporting)
        # Note: final_w sums to 1 over (n_int + n_ext) units.
        weights_external = final_w[n_int:]
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=weights_external,
            weights_external=weights_external * n_pool,
            energy_distance=loss_cov.item(),
            estimation_time=time.time() - start_time
        )