import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structures import SplitData, EstimationResult
from .base import BaseEstimator


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
                 prog_hidden_dim=64, device=None):
        """
        Args:
            lamb: Regularization strength for prognostic term (higher = more outcome guidance)
            lr_weights: Learning rate for weight optimization
            lr_prog: Learning rate for prognostic model training
            n_iter_weights: Number of iterations for weight optimization
            n_iter_prog: Number of iterations for prognostic model training
            prog_hidden_dim: Hidden dimension for prognostic neural network
            device: Torch device (defaults to CUDA if available)
        """
        super().__init__()
        self.lamb = lamb
        self.lr_weights = lr_weights
        self.lr_prog = lr_prog
        self.n_iter_weights = n_iter_weights
        self.n_iter_prog = n_iter_prog
        self.prog_hidden_dim = prog_hidden_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _energy_distance(self, X, Y, w1=None, w2=None):
        """
        Compute energy distance between two weighted point clouds.
        
        ED(P, Q) = 2*E[d(X,Y)] - E[d(X,X')] - E[d(Y,Y')]
        where X~P, Y~Q, and d is L1 distance.
        """
        if w1 is None:
            w1 = torch.ones(X.shape[0], device=X.device) / X.shape[0]
        if w2 is None:
            w2 = torch.ones(Y.shape[0], device=Y.device) / Y.shape[0]
        
        d_XY = torch.cdist(X, Y, p=1)  # [n_X, n_Y]
        d_XX = torch.cdist(X, X, p=1)  # [n_X, n_X]
        d_YY = torch.cdist(Y, Y, p=1)  # [n_Y, n_Y]
        
        term1 = 2*torch.sum(w1.unsqueeze(1) * w2.unsqueeze(0) * d_XY)
        term2 = torch.sum(w1.unsqueeze(1) * w1.unsqueeze(0) * d_XX)
        term3 = torch.sum(w2.unsqueeze(1) * w2.unsqueeze(0) * d_YY)
        
        return term1 - term2 - term3
    
    def _train_prognostic_model(self, X_ext, Y_ext):
        # TODO: Change to RF, add some regularization (e.g. k-fold), ...
        """
        Train prognostic model on external data using energy score loss.
        
        The energy score is a proper scoring rule:
        ES = E[|y_pred - y|] - 0.5 * E[|y_pred - y_pred'|]
        
        Minimizing this trains the model to predict the conditional distribution.
        """
        class PrognosticModel(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            
            def forward(self, x):
                return self.net(x).view(-1, 1)
        
        model = PrognosticModel(X_ext.shape[1], self.prog_hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_prog)
        
        for _ in range(self.n_iter_prog):
            optimizer.zero_grad()
            
            # Two forward passes for energy score
            y_pred1 = model(X_ext)
            y_pred2 = model(X_ext)
            
            # Energy score: E|y_pred - y| - 0.5 * E|y_pred1 - y_pred2|
            loss_es = (torch.mean(torch.abs(y_pred1.squeeze() - Y_ext.squeeze())) - 
                      0.5 * torch.mean(torch.abs(y_pred1.squeeze() - y_pred2.squeeze())))
            
            loss_es.backward()
            optimizer.step()
        
        model.eval()
        return model
    
    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        
        # Convert to tensors
        X_treat = torch.tensor(data.X_treat, dtype=torch.float32, device=self.device)
        X_control_int = torch.tensor(data.X_control_int, dtype=torch.float32, device=self.device)
        X_external = torch.tensor(data.X_external, dtype=torch.float32, device=self.device)
        Y_external = torch.tensor(data.Y_external, dtype=torch.float32, device=self.device).view(-1, 1)
        
        X_pooled_control = torch.cat([X_control_int, X_external], dim=0)
        Y_pooled_control = np.concatenate([data.Y_control_int, data.Y_external], axis=0)
        
        # Step 1: Train prognostic model on external data
        prog_model = self._train_prognostic_model(X_external, Y_external)
        
        # Step 2: Get prognostic scores for all control units
        with torch.no_grad():
            m_control_int = prog_model(X_control_int)
            m_external = prog_model(X_external)
        
        # Step 3: Optimize weights to balance covariates and prognostic scores
        n_pool = X_pooled_control.shape[0]
        logits = torch.zeros(n_pool, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([logits], lr=self.lr_weights)
        
        prev_loss = float('inf')
        for epoch in range(self.n_iter_weights):
            optimizer.zero_grad()
            
            w = F.softmax(logits, dim=0)
            w_ext = w[n_int:]
            
            # Loss 1: Covariate balance (treatment vs weighted pooled control)
            loss_cov = self._energy_distance(X_treat, X_pooled_control, None, w)
            
            # Loss 2: Prognostic balance (RCT control vs weighted external)
            loss_prog = self._energy_distance(m_control_int, m_external, None, w_ext)
            
            # Combined loss
            loss = loss_cov + self.lamb * loss_prog
            
            loss.backward()
            optimizer.step()
            
            # Early stopping if converged
            if abs(prev_loss - loss.item()) < 1e-6:
                break
            prev_loss = loss.item()
        
        # Step 4: Compute ATE with final weights
        final_w = F.softmax(logits, dim=0).detach().cpu().numpy()
        
        y1_mean = np.mean(data.Y_treat)
        y0_weighted_mean = np.average(Y_pooled_control, weights=final_w)
        ate = y1_mean - y0_weighted_mean
        
        # Extract external weights for result
        w_ext_final = final_w[n_int:]
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=w_ext_final,
            weights_external=w_ext_final * n_pool,
            energy_distance=self._energy_distance(X_treat, X_pooled_control, None, torch.tensor(final_w, device=self.device))
        )