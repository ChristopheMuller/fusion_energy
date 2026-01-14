import numpy as np
import torch
import torch.nn.functional as F

class EnergyWeighting:
    """
    Learns weights for a combined pool (Internal + External) to minimize 
    Energy Distance to the Target distribution.
    
    - No sampling or subset selection.
    - Treats Internal and External units equally (concatenated).
    - Returns weights for the full combined dataset.
    """
    def __init__(self, lr=0.01, n_iter=500, device=None, **kwargs):
        self.lr = lr
        self.n_iter = n_iter
        self.weights_ = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X_target, X_internal, X_external):
        # 1. Prepare Data
        # Combine Internal and External into a single Control pool
        X_control_np = np.vstack([X_internal, X_external])
        
        # Move to tensor
        X_t = torch.tensor(X_target, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(X_control_np, dtype=torch.float32, device=self.device)
        
        n_c = X_c.shape[0] # Number of control units
        n_t = X_t.shape[0] # Number of target units
        
        # 2. Pre-compute Distance Matrices
        # We need to minimize Energy Distance: 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
        # X is Weighted Control, Y is Uniform Target.
        # The term E[|Y-Y'|] (Target-Target) is constant w.r.t weights, so we ignore it.
        
        # Term A: Control vs Target (Interaction)
        # Shape: (n_c, n_t) -> sum over dim 1 -> (n_c,)
        d_ct = torch.cdist(X_c, X_t)
        d_ct_sum = d_ct.sum(dim=1) 
        
        # Term B: Control vs Control (Self-Interaction)
        # Shape: (n_c, n_c)
        d_cc = torch.cdist(X_c, X_c)

        # 3. Optimization Loop
        # We optimize logits to ensure weights sum to 1 via Softmax
        logits = torch.zeros(n_c, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([logits], lr=self.lr)
        
        for _ in range(self.n_iter):
            opt.zero_grad()
            w = F.softmax(logits, dim=0)
            
            # --- Energy Distance Objective ---
            # We want to minimize: 2 * E[d(weighted_control, target)] - E[d(weighted_control, weighted_control)]
            
            # Term 1: Interaction (2 * w^T * D_ct * 1/n_t)
            # We average over Target (n_t), but sum over Control weights (w)
            term1 = 2 * torch.dot(w, d_ct_sum) / n_t
            
            # Term 2: Self-Interaction (w^T * D_cc * w)
            # This penalizes clustering all weights on a single point (encourages diversity)
            term2 = torch.dot(w, torch.mv(d_cc, w))
            
            # Energy Distance = Term 1 - Term 2
            loss = term1 - term2
            
            loss.backward()
            opt.step()
            
        self.weights_ = F.softmax(logits, dim=0).detach().cpu().numpy()
        return self

    def sample(self, X_target, X_internal, X_external, Y_internal=None, Y_external=None):

        if self.weights_ is None:
            raise ValueError("Must call fit() before sample()")
        
        if Y_internal is not None and Y_external is not None:
            return None, None, self.weights_
        
        return None, self.weights_