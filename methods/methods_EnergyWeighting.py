import numpy as np
import torch
import torch.nn.functional as F

class EnergyAugmenter_Weighting:
    """
    Learns weights for a combined pool (Internal + External) to minimize 
    Energy Distance to the Target distribution.
    
    - No sampling or subset selection.
    - Treats Internal and External units equally (concatenated).
    - Returns weights for the full combined dataset.
    """
    def __init__(self, n_sampled=None, lr=0.01, n_iter=500, device=None, **kwargs):
        # n_sampled is unused for weighting logic but kept for interface compatibility
        self.n_sampled = n_sampled
        self.lr = lr
        self.n_iter = n_iter
        self.weights_ = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X_target, X_control, X_external):
        # 1. Prepare Data
        # Combine Internal and External into a single Control pool
        # For the experiment setup, X_control might be empty, so we weight X_external
        X_control_np = np.vstack([X_control, X_external])
        
        # Move to tensor
        X_t = torch.tensor(X_target, dtype=torch.float32, device=self.device)
        X_c = torch.tensor(X_control_np, dtype=torch.float32, device=self.device)
        
        n_c = X_c.shape[0] # Number of control units
        n_t = X_t.shape[0] # Number of target units
        
        if n_c == 0:
            self.weights_ = np.array([])
            return self
        
        # 2. Pre-compute Distance Matrices
        # We need to minimize Energy Distance: 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
        # X is Weighted Control, Y is Uniform Target.
        
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
        
        # Reconstruct the pool used in fit
        X_pool = np.vstack([X_internal, X_external])
        
        # Return the pool and its weights
        # If Ys are provided, stack them too
        if Y_internal is not None and Y_external is not None:
            Y_pool = np.concatenate([Y_internal, Y_external])
            return None, None, self.weights_
        
        return None, self.weights_
    
    def get_internal_weights(self):
        return self.weights_