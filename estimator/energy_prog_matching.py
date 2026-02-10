import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.ensemble import RandomForestRegressor

from structures import SplitData, EstimationResult
from .base import BaseEstimator
from .prognostic_energy_weighting import PrognosticEnergyWeightingEstimator 

from metrics import compute_batch_energy 

class PrognosticEnergyMatchingEstimator(BaseEstimator):
    """
    Selects a subset of external controls to minimize a dual objective:
    1. Covariate Balance: Energy(Pooled Control, Treatment)
    2. Prognostic Balance: Energy(Selected External Prognosis, RCT Control Prognosis)
    
    Strategy:
    1. Run PrognosticEnergyWeightingEstimator to get optimal soft weights.
    2. Use soft weights as sampling probabilities to generate K candidate subsets.
    3. Select the specific subset that minimizes the dual discrete energy loss.
    """
    
    def __init__(self, n_external: int = None, k_best=100, lamb=0.1, 
                 lr_weights=0.01, n_iter_weights=600, 
                 device=None, verbose=False):
        super().__init__(n_external=n_external)
        self.k_best = k_best
        self.lamb = lamb
        
        self.lr_weights = lr_weights
        self.n_iter_weights = n_iter_weights
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def _train_prognostic_model(self, X_ext, Y_ext):
        # Helper to train the prognostic model (Random Forest)
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
        
        # 1. Determine matching size
        if n_external is not None:
            target_n = n_external
        elif self.n_external is not None:
            target_n = int(self.n_external)
        else:
            target_n = data.target_n_aug
            
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]

        # 2. Step 1: Get Soft Weights via PrognosticEnergyWeightingEstimator
        if self.verbose:
            print(">>> Step 1: Optimizing soft weights...")
            
        weighting_est = PrognosticEnergyWeightingEstimator(
            lamb=self.lamb,
            lr_weights=self.lr_weights,
            n_iter_weights=self.n_iter_weights,
            device=self.device,
            verbose=self.verbose
        )
        # We assume weighting_est returns continuous weights for external data in .weights_continuous
        weighting_result = weighting_est.estimate(data)
        
        # Convert weights to probabilities
        soft_weights = weighting_result.weights_continuous # (n_ext,)
        probs = np.abs(soft_weights) 
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones(n_ext) / n_ext
        else:
            probs = probs / probs_sum

        # 3. Step 2: Prepare Data for Selection
        X_t = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_e = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)
        Y_ext_t = torch.tensor(data.Y_external, dtype=torch.float32, device=self.device).view(-1, 1)

        # Train Prognostic Model (Need this for the selection metric)
        # TODO: change this by extracting it from the weighting estimator to avoid redundant training. For now, we retrain for simplicity.
        prog_model = self._train_prognostic_model(X_e, Y_ext_t)
        
        with torch.no_grad():
            m_c = prog_model(X_c)
            m_e = prog_model(X_e)

        # 4. Step 3: Stochastic Selection (Dual Loss)
        if self.verbose:
            print(f">>> Step 2: Selecting best {target_n} samples from {self.k_best} candidates...")
            
        best_indices, min_loss, energy_cov, energy_prog = self._select_best_sample_dual(
            X_t, X_c, X_e, 
            m_c, m_e, 
            probs, target_n
        )
        
        # 5. Construct Result
        final_w_ext = np.zeros(n_ext)
        final_w_ext[best_indices] = 1.0 # Binary weights
        
        # Calculate ATE
        # Control Outcome = Weighted average of Internal (weight 1) and Selected External (weight 1)
        # Denominator = n_int + target_n
        y_control_pooled = np.concatenate([data.Y_control_int, data.Y_external[best_indices]])
        y0_mean = np.mean(y_control_pooled)
        y1_mean = np.mean(data.Y_treat)
        ate = y1_mean - y0_mean

        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=probs, # Return the soft probs as 'continuous' info
            weights_external=final_w_ext, # The actual binary match
            energy_distance=energy_cov, # Primary metric
            estimation_time=time.time() - start_time
        )

    def _select_best_sample_dual(self, X_t, X_c, X_e, m_c, m_e, probs, n_sampled):
        """
        Samples k_best subsets and minimizes:
        Loss = Energy(Pooled_Cov, Treat) + lambda * Energy(Ext_Prog, Int_Prog)
        """
        # A. Sample Indices
        n_ext = len(probs)
        pool_idx = np.arange(n_ext)
        
        batch_indices_list = []
        for _ in range(self.k_best):
            try:
                idx = np.random.choice(pool_idx, size=n_sampled, replace=False, p=probs)
            except ValueError:
                # Fallback if p contains NaNs or sum != 1 (numerical stability)
                idx = np.random.choice(pool_idx, size=n_sampled, replace=False)
            batch_indices_list.append(idx)
            
        batch_idx_tensor = torch.tensor(np.array(batch_indices_list), device=self.device, dtype=torch.long)
        
        # B. Prepare Batches
        # 1. Covariates: Batch of External Candidates
        X_e_batch = X_e[batch_idx_tensor] # (k, n_sampled, dim)
        
        # 2. Prognostic Scores: Batch of External Candidates
        m_e_batch = m_e[batch_idx_tensor] # (k, n_sampled, 1)

        # C. Compute Energies (Batch)
        
        # Term 1: Covariate Balance
        # Source = Batch External, Internal = Fixed Control, Target = Fixed Treat
        energy_cov = compute_batch_energy(
            X_source_batch=X_e_batch,
            X_target=X_t,
            X_internal=X_c
        ) # Returns (k,)
        
        # Term 2: Prognostic Balance
        # Source = Batch External Scores, Target = Fixed Control Scores
        # Note: No 'Internal' here. We compare Ext directly to Int.
        energy_prog = compute_batch_energy(
            X_source_batch=m_e_batch,
            X_target=m_c, 
            X_internal=None 
        ) # Returns (k,)
        
        # D. Scaling
        # To combine them fairly, we normalize by the first batch's magnitude 
        # (or 1.0 to avoid division by zero). 
        # This mirrors the adaptive scaling in the weighting estimator.
        s_cov = max(energy_cov[0].item(), 1e-6)
        s_prog = max(energy_prog[0].item(), 1e-6)
        
        total_loss = (energy_cov / s_cov) + self.lamb * (energy_prog / s_prog)
        
        # E. Pick Best
        best_k = torch.argmin(total_loss).item()
        
        return (
            batch_indices_list[best_k], 
            total_loss[best_k].item(),
            energy_cov[best_k].item(),
            energy_prog[best_k].item()
        )