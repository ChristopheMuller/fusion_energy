import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.ensemble import RandomForestRegressor

from structures import SplitData, EstimationResult
from .base import BaseEstimator
from metrics import compute_batch_energy

class PrognosticEnergyMatchingEstimator(BaseEstimator):
    """
    Selects a subset of external controls to minimize a dual objective for the POOLED control arm.
    
    The optimization explicitly models the control arm as a mixture:
        P_control = (1 - beta) * Internal_Fixed + beta * External_Weighted
        
    Where beta is fixed by the target matching ratio: 
        beta = n_match / (n_match + n_internal)
    """
    
    def __init__(self, n_external: int = None, k_best=100, lamb=0.1, 
                 lr=0.05, n_iter=500, 
                 device=None, verbose=False):
        super().__init__(n_external=n_external)
        self.k_best = k_best
        self.lamb = lamb
        self.lr = lr
        self.n_iter = n_iter
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.prog_model = None # Stateful storage for the model

    def _train_prognostic_model(self, X_ext, Y_ext):
        # Helper to train the prognostic model (Random Forest)
        X_np = X_ext.detach().cpu().numpy()
        Y_np = Y_ext.detach().cpu().numpy().ravel()
        
        rf = RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=1)
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
        
        # 1. Determine Sample Sizes & Beta
        if n_external is not None:
            target_n = n_external
        elif self.n_external is not None:
            target_n = int(self.n_external)
        else:
            target_n = data.target_n_aug
            
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        n_treat = data.X_treat.shape[0]
        
        # Beta is the mixing proportion of the external data in the final control arm
        beta = target_n / (target_n + n_int)

        # 2. Data to Tensor
        X_t = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)
        Y_e = torch.tensor(data.Y_external, dtype=torch.float32, device=self.device).view(-1, 1)

        # 3. Prognostic Model
        # Train and store
        self.prog_model = self._train_prognostic_model(X_e, Y_e)
        
        with torch.no_grad():
            m_c = self.prog_model(X_c) 
            m_e = self.prog_model(X_e)

        # 4. Step 1: Optimize Soft Weights (Mixture Aware)
        # We optimize w_ext so that Mixture(Fixed Internal, Weighted External) matches Target
        if self.verbose:
            print(f">>> Optimizing Mixture Weights (beta={beta:.3f})...")

        probs = self._optimize_mixture_weights(
            X_t, X_c, X_e, 
            m_c, m_e,
            beta, n_iter=self.n_iter
        )

        # 5. Step 2: Stochastic Selection
        # We sample discrete subsets based on 'probs' and pick the best one
        if self.verbose:
            print(f">>> Selecting best {target_n} samples from {self.k_best} candidates...")
            
        best_indices, min_loss, energy_cov, energy_prog = self._select_best_sample_dual(
            X_t, X_c, X_e, 
            m_c, m_e, 
            probs, target_n
        )
        
        # 6. Construct Result
        final_w_ext = np.zeros(n_ext)
        final_w_ext[best_indices] = 1.0
        
        y_control_pooled = np.concatenate([data.Y_control_int, data.Y_external[best_indices]])
        ate = np.mean(data.Y_treat) - np.mean(y_control_pooled)

        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=probs,
            weights_external=final_w_ext,
            energy_distance=energy_cov,
            estimation_time=time.time() - start_time,
            energy_distance_prognostic=energy_prog
        )

    def _optimize_mixture_weights(self, X_t, X_c, X_e, m_c, m_e, beta, n_iter):
        """
        Optimizes logits for X_e to minimize:
        Loss = Energy( (1-B)Xc + B*Xe(w), Xt ) + lambda * Energy( Xe(w), Xc ) [Prognostic Space]
        """
        n_ext = X_e.shape[0]
        n_t = X_t.shape[0]
        n_c = X_c.shape[0]
        
        # --- A. Precompute Constant Terms (Linear in w) ---
        with torch.no_grad():
            # 1. Covariate Terms
            d_et_mean = torch.cdist(X_e, X_t).mean(dim=1) 
            d_ec_mean = torch.cdist(X_e, X_c).mean(dim=1)
            d_ee = torch.cdist(X_e, X_e)

            # 2. Prognostic Terms
            d_m_ec_mean = torch.cdist(m_e, m_c).mean(dim=1)
            d_m_ee = torch.cdist(m_e, m_e)

            # Scaling Factors (Calculate FULL loss at uniform weights)
            w_unif = torch.ones(n_ext, device=self.device) / n_ext
            
            # --- CORRECTED s_cov CALCULATION ---
            # We mirror the exact formula used in the loop
            term_cross_et_unif = torch.dot(w_unif, d_et_mean)
            term_self_ee_unif  = torch.dot(w_unif, torch.mv(d_ee, w_unif))
            term_cross_ec_unif = torch.dot(w_unif, d_ec_mean)
            
            base_cov_loss = (2 * beta * term_cross_et_unif) \
                           - (beta**2 * term_self_ee_unif) \
                           - (2 * beta * (1 - beta) * term_cross_ec_unif)
            
            s_cov = max(abs(base_cov_loss.item()), 1e-6)
            
            # --- s_prog CALCULATION (Standard Energy) ---
            prog_cross_unif = torch.dot(w_unif, d_m_ec_mean)
            prog_self_unif  = torch.dot(w_unif, torch.mv(d_m_ee, w_unif))
            
            base_prog_loss = 2 * prog_cross_unif - prog_self_unif
            s_prog = max(abs(base_prog_loss.item()), 1e-6)
        
        # --- B. Optimization Loop ---
        logits = torch.zeros(n_ext, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([logits], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iter, eta_min=1e-5
            )
        prev_loss = float('inf')
        
        for i in range(n_iter):
            optimizer.zero_grad()
            w = F.softmax(logits, dim=0)
            
            # --- Term 1: Covariate Balance (Mixture) ---
            # Energy(P, Q) parts dependent on w:
            # + 2 * beta * E[d(Xe, Xt)]
            # - beta^2 * E[d(Xe, Xe)]
            # - 2 * beta * (1-beta) * E[d(Xe, Xc)]
            
            term_cross_et = torch.dot(w, d_et_mean)
            term_self_ee  = torch.dot(w, torch.mv(d_ee, w))
            term_cross_ec = torch.dot(w, d_ec_mean)
            
            loss_cov = (2 * beta * term_cross_et) - (beta**2 * term_self_ee) - (2 * beta * (1 - beta) * term_cross_ec)
            
            # --- Term 2: Prognostic Balance ---
            # Standard Energy(Xe, Xc) in prognostic space
            # + 2 * E[d(m_e, m_c)]
            # - E[d(m_e, m_e)]
            
            prog_cross = torch.dot(w, d_m_ec_mean)
            prog_self  = torch.dot(w, torch.mv(d_m_ee, w))
            
            loss_prog = 2 * prog_cross - prog_self
            
            # --- Total Loss ---
            loss = (loss_cov / s_cov) + self.lamb * (loss_prog / s_prog)

            if abs(prev_loss - loss.item()) < 1e-7 and i > 20: 
                if self.verbose:
                    print(f"  > Converged at iter {i} with Loss: {loss.item():.6f}")
                break

            prev_loss = loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if self.verbose and (i % 20 == 0 or i == n_iter - 1):
                print(f"Iter {i+1}/{n_iter} - Loss: {loss.item():.4f} (Cov: {(loss_cov/s_cov).item():.4f}, Prog: {(loss_prog/s_prog).item():.4f}), LR: {scheduler.get_last_lr()[0]:.5f}")
            
            
        return F.softmax(logits, dim=0).detach().cpu().numpy()

    def _select_best_sample_dual(self, X_t, X_c, X_e, m_c, m_e, probs, n_sampled):
        """
        Samples k_best subsets and minimizes discrete Dual Loss.
        """
        # 1. Sample Indices (Vectorized)
        probs_tensor = torch.tensor(probs, device=self.device, dtype=torch.float32)
        batch_idx_tensor = torch.multinomial(
            probs_tensor.expand(self.k_best, -1),
            num_samples=n_sampled,
            replacement=False
        )
        
        # 2. Prepare Batches
        X_e_batch = X_e[batch_idx_tensor] # (k, n, dim)
        m_e_batch = m_e[batch_idx_tensor] # (k, n, 1)

        # 3. Compute Energies using Batch Util
        
        # Covariate Energy (Pooled Control vs Treat)
        # Internal is FIXED, Source is BATCHED
        energy_cov = compute_batch_energy(
            X_source_batch=X_e_batch,
            X_target=X_t,
            X_internal=X_c # Crucial: Pass Internal to compute Mixture Energy
        )
        
        # Prognostic Energy (External Prognosis vs Internal Prognosis)
        # Direct comparison, no internal component in "Source"
        energy_prog = compute_batch_energy(
            X_source_batch=m_e_batch,
            X_target=m_c, 
            X_internal=None 
        )
        
        # 4. Combine
        s_cov = max(energy_cov.mean().item(), 1e-6)
        s_prog = max(energy_prog.mean().item(), 1e-6)
        
        total_loss = (energy_cov / s_cov) + self.lamb * (energy_prog / s_prog)
        
        best_k = torch.argmin(total_loss).item()
        
        return (
            batch_idx_tensor[best_k].cpu().numpy(),
            total_loss[best_k].item(),
            energy_cov[best_k].item(),
            energy_prog[best_k].item()
        )