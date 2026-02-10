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
                return torch.tensor(preds, dtype=torch.float32, device=self.device).view(-1, 1)
                
        return RFWrapper(rf, self.device)
    
    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
        
        # --- 1. Data Prep ---
        X_treat = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_control_int = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_external = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)
        Y_external = torch.tensor(data.Y_external, dtype=torch.float32, device=self.device).view(-1, 1)
        
        n_int = X_control_int.shape[0]
        n_ext = X_external.shape[0]
        n_treat = X_treat.shape[0]
        
        X_pooled_control = torch.cat([X_control_int, X_external], dim=0)
        Y_pooled_control = np.concatenate([data.Y_control_int, data.Y_external], axis=0)
        n_pool = n_int + n_ext

        # --- 2. Prognostic Model ---
        prog_model = self._train_prognostic_model(X_external, Y_external)
        with torch.no_grad():
            m_control_int = prog_model(X_control_int) # Target for prognostic term
            m_external = prog_model(X_external)       # Source for prognostic term

        # --- 3. Precompute Distances & Linear Terms ---
        
        # A. Covariate Balance Terms (Target: Treat, Source: Pooled Control)
        # We need d(Pooled, Treat) and d(Pooled, Pooled)
        with torch.no_grad():
            d_pool_treat = torch.cdist(X_pooled_control, X_treat, p=2) # (n_pool, n_treat)
            cov_linear_term = (2.0 / n_treat) * d_pool_treat.sum(dim=1) # (n_pool,)
            
            d_pool_pool = torch.cdist(X_pooled_control, X_pooled_control, p=2) # (n_pool, n_pool)
            
            w_unif = torch.ones(n_pool, device=self.device) / n_pool
            base_cov_loss = (torch.dot(w_unif, cov_linear_term) - torch.dot(w_unif, torch.mv(d_pool_pool, w_unif))).item()
            s_cov = max(abs(base_cov_loss), 1e-6)

        # B. Prognostic Balance Terms (Target: Control Int, Source: External)
        # We need d(External, Control_Int) and d(External, External)
        # Note: We only weight the 'External' part for this specific loss term
        with torch.no_grad():
            d_m_ext_int = torch.cdist(m_external, m_control_int, p=2) # (n_ext, n_int)
            # Pre-average over internal control: 2 * sum(d) / n_int
            prog_linear_term = (2.0 / n_int) * d_m_ext_int.sum(dim=1) # (n_ext,)
            
            d_m_ext_ext = torch.cdist(m_external, m_external, p=2) # (n_ext, n_ext)
            
            # Constants for scaling
            w_ext_unif = torch.ones(n_ext, device=self.device) / n_ext
            base_prog_loss = (torch.dot(w_ext_unif, prog_linear_term) - torch.dot(w_ext_unif, torch.mv(d_m_ext_ext, w_ext_unif))).item()
            s_prog = max(abs(base_prog_loss), 1e-6)

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
            
            # Softmax over ALL pooled controls
            w = F.softmax(logits, dim=0)
            
            # 1. Covariate Loss: 2<w, d_XT> - <w, d_XX w>
            # Uses the precomputed linear term for speed
            loss_cov_term1 = torch.dot(w, cov_linear_term)
            loss_cov_term2 = torch.dot(w, torch.mv(d_pool_pool, w))
            loss_cov = loss_cov_term1 - loss_cov_term2
            
            # 2. Prognostic Loss: Affects only external weights part of w
            w_ext = w[n_int:]
            # Normalize w_ext so it sums to 1 for the prognostic comparison
            w_ext_sum = w_ext.sum()
            w_ext_norm = w_ext / (w_ext_sum + 1e-10)
            
            loss_prog_term1 = torch.dot(w_ext_norm, prog_linear_term)
            loss_prog_term2 = torch.dot(w_ext_norm, torch.mv(d_m_ext_ext, w_ext_norm))
            loss_prog = loss_prog_term1 - loss_prog_term2
            
            # Combine
            total_loss = (loss_cov / s_cov) + self.lamb * (loss_prog / s_prog)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            current_loss = total_loss.item()
            
            if self.verbose and (epoch % 100 == 0):
                print(f"Epoch {epoch:4d} | Total: {total_loss.item():.6f} | "
                      f"Cov: {loss_cov.item():.6f} | Prog: {loss_prog.item():.6f}")

            # --- EARLY STOPPING RESTORED ---
            # We wait until at least 25% of iterations are done to avoid stopping 
            # during the initial volatile phase of OneCycleLR warmup.
            if abs(prev_loss - current_loss) < 1e-8 and epoch > self.n_iter_weights // 4:
                if self.verbose:
                    print(f"Converged at epoch {epoch} with loss {current_loss:.6f}")
                break
            prev_loss = current_loss

        # --- 5. Result ---
        final_w = F.softmax(logits, dim=0).detach().cpu().numpy()
        
        # ATE Calculation
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