import torch
import torch.nn.functional as F
import numpy as np

def optimise_soft_weights(
    X_source: torch.Tensor,
    X_target: torch.Tensor,
    X_internal: torch.Tensor = None,
    target_n_aug: int = None,
    lr: float = 0.05,
    n_iter: int = 1000,
    dist_st: torch.Tensor = None,
    dist_ss: torch.Tensor = None,
    dist_is: torch.Tensor = None,
    dist_st_sum: torch.Tensor = None,
    dist_is_sum: torch.Tensor = None
):
    """
    Optimises soft weights (logits) for X_source to minimize Energy Distance.
    
    Scenarios:
    1. Augmentation / Mixture (X_internal provided):
       Target Distribution Q = X_target
       Source Distribution P = Mixture of (Fixed X_internal) and (Weighted X_source).
       P = (1-beta) * Internal + beta * Source_Weighted
       
       Beta is determined by `target_n_aug` relative to `X_internal` size.
       
    2. Direct Matching (X_internal is None):
       Target Distribution Q = X_target
       Source Distribution P = Weighted X_source
       (Equivalent to beta=1.0)
    
    Args:
        X_source: Tensor (n_s, dim) - The candidate pool (e.g., External data).
        X_target: Tensor (n_t, dim) - The target population (e.g., RCT Treatment).
        X_internal: Tensor (n_i, dim) - Optional. Fixed component of source (e.g., RCT Control).
        target_n_aug: int - Required if X_internal is present. The effective sample size of source.
        lr: Learning rate.
        n_iter: Number of iterations.
        dist_st: Optional precomputed Source -> Target distance matrix.
        dist_ss: Optional precomputed Source -> Source distance matrix.
        dist_is: Optional precomputed Internal -> Source distance matrix.
        
    Returns:
        logits: Tensor (n_s,) - Unnormalized log-probabilities.
    """
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    
    # Precompute distances
    # d_st: Source -> Target
    if dist_st_sum is not None:
        d_st_sum = dist_st_sum
    elif dist_st is not None:
        d_st_sum = dist_st.sum(dim=1)
    else:
        d_st_sum = torch.cdist(X_source, X_target).sum(dim=1)
    
    # d_ss: Source -> Source
    if dist_ss is not None:
        d_ss = dist_ss
    else:
        d_ss = torch.cdist(X_source, X_source)
    
    if X_internal is not None:
        if target_n_aug is None:
             raise ValueError("target_n_aug required when X_internal is provided")
             
        n_internal = X_internal.shape[0]
        total_n = n_internal + target_n_aug
        
        # Avoid division by zero
        beta = target_n_aug / total_n if total_n > 0 else 0.5
        
        # d_is: Internal -> Source
        if dist_is_sum is not None:
            d_is_sum = dist_is_sum
        elif dist_is is not None:
            d_is_sum = dist_is.sum(dim=0)
        else:
            d_is_sum = torch.cdist(X_internal, X_source).sum(dim=0)
    else:
        beta = 1.0
        n_internal = 0
        d_is_sum = None
        
    # optimisation
    logits = torch.zeros(n_source, requires_grad=True, device=X_source.device)
    opt = torch.optim.Adam([logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr*0.01)
    
    prev_loss = float('inf')
    for _ in range(n_iter):
        opt.zero_grad()
        w = F.softmax(logits, dim=0)
        
        # Term 1: Cross term (Source -> Target)
        # Contribution: 2 * beta * E[d(S_w, T)]
        term1 = torch.dot(w, d_st_sum) / n_target if n_target > 0 else 0.0
        
        # Term 2: Self term (Source part)
        # Contribution: - beta^2 * E[d(S_w, S_w)]
        term2_ss = torch.dot(w, torch.mv(d_ss, w))
        
        if X_internal is not None:
            # Term 3: Cross Internal-Source
            # Contribution: - 2 * beta * (1-beta) * E[d(S_w, I)]
            term2_is = torch.dot(w, d_is_sum) / n_internal if n_internal > 0 else 0.0
            
            # Energy Loss (Variable parts)
            loss = (2 * beta * term1) - (beta**2 * term2_ss + 2 * beta * (1 - beta) * term2_is)
        else:
            # Pure Energy: 2 E[d(S, T)] - E[d(S, S)]
            loss = 2 * term1 - term2_ss
            
        loss.backward()
        opt.step()
        scheduler.step()
        if abs(prev_loss - loss.item()) < 1e-8:
            break
        prev_loss = loss.item()
        
    return logits.detach()


def compute_batch_energy(
    X_source_batch: torch.Tensor, 
    X_target: torch.Tensor,       
    X_internal: torch.Tensor = None,
    sum_it: torch.Tensor = None,
    sum_ii: torch.Tensor = None
):
    """
    Computes Energy Distance for a batch of source subsets.
    
    Args:
        X_source_batch: (k, n_s, dim) - Batch of subset candidates.
        X_target: (n_t, dim) - Target population.
        X_internal: (n_i, dim) - Optional fixed source component.
        sum_it: (Optional) Precomputed sum of distances between X_internal and X_target.
        sum_ii: (Optional) Precomputed sum of distances between X_internal and X_internal.
        
    Returns:
        energies: (k,) - Energy distance for each batch.
    """
    k, n_s, _ = X_source_batch.shape
    n_t = X_target.shape[0]
    
    # Expand Target for batch: (1, n_t, dim) -> (k, n_t, dim)
    X_t_exp = X_target.unsqueeze(0).expand(k, -1, -1)
    
    # 1. Cross Term: Source <-> Target
    d_st = torch.cdist(X_source_batch, X_t_exp) # (k, n_s, n_t)
    sum_st = d_st.sum(dim=(1, 2)) # (k,)
    
    # 2. Self Term: Source <-> Source
    d_ss = torch.cdist(X_source_batch, X_source_batch) # (k, n_s, n_s)
    sum_ss = d_ss.sum(dim=(1, 2)) # (k,)
    
    if X_internal is not None:
        n_i = X_internal.shape[0]
        n_pool = n_i + n_s
        
        # Constant Terms (Internal <-> Target, Internal <-> Internal)
        # These are scalars, computed once.
        if sum_it is None:
            sum_it = torch.cdist(X_internal, X_target).sum()
        if sum_ii is None:
            sum_ii = torch.cdist(X_internal, X_internal).sum()
        
        # Cross Internal <-> Source
        # X_internal: (n_i, dim) -> (1, n_i, dim) -> (k, n_i, dim)
        X_i_exp = X_internal.unsqueeze(0).expand(k, -1, -1)
        d_is = torch.cdist(X_source_batch, X_i_exp) # (k, n_s, n_i)
        sum_is = d_is.sum(dim=(1, 2)) # (k,)
        
        # Total Cross: (Sum_IT + Sum_ST) / (N_pool * N_t)
        term_cross = (sum_it + sum_st) / (n_pool * n_t)
        
        # Total Self: (Sum_II + Sum_SS + 2*Sum_IS) / (N_pool^2)
        term_self = (sum_ii + sum_ss + 2 * sum_is) / (n_pool**2)
        
        return 2 * term_cross - term_self
        
    else:
        # Pure Energy: 2 E[d(S, T)] - E[d(S, S)]
        term_cross = sum_st / (n_s * n_t)
        term_self = sum_ss / (n_s**2)
        
        return 2 * term_cross - term_self


def compute_weighted_energy(
    X_target: np.ndarray,
    X_internal: np.ndarray,
    X_external: np.ndarray,
    weights_external: np.ndarray,
    device=None
) -> float:
    """
    Computes the Energy Distance between:
    1. Target Distribution (X_target) - Uniform weights
    2. Source Distribution (Pooled) - Mixture of:
       - X_internal (Uniform weights = 1.0)
       - X_external (Provided weights)
       
    Args:
        X_target: (n_t, dim)
        X_internal: (n_i, dim)
        X_external: (n_e, dim)
        weights_external: (n_e,) - Weights for external units.
        
    Returns:
        float: The scalar Energy Distance.
    """
    if device is None:
        device = torch.device("cpu") # Default to CPU for evaluation to save GPU memory/transfer overhead if not needed

    # Convert to Tensor
    Xt = torch.as_tensor(X_target, dtype=torch.float32, device=device)
    Xi = torch.as_tensor(X_internal, dtype=torch.float32, device=device)
    Xe = torch.as_tensor(X_external, dtype=torch.float32, device=device)
    w_ext = torch.as_tensor(weights_external, dtype=torch.float32, device=device)
    
    n_t = Xt.shape[0]
    n_i = Xi.shape[0]
    
    # Normalize weights
    # Internal units have weight 1.0
    w_int_sum = float(n_i)
    w_ext_sum = w_ext.sum().item()
    total_weight_source = w_int_sum + w_ext_sum
    
    if total_weight_source == 0:
        return float('inf')

    # --- Term 1: 2 * E[d(Source, Target)] ---
    # Split into Internal-Target and External-Target
    
    # 1a. Internal <-> Target
    d_it = torch.cdist(Xi, Xt) # (n_i, n_t)
    sum_it = d_it.sum()
    
    # 1b. External <-> Target (Weighted)
    d_et = torch.cdist(Xe, Xt) # (n_e, n_t)
    # Sum over target (dim 1), then dot with weights (dim 0)
    sum_et_per_ext = d_et.sum(dim=1) # (n_e,)
    sum_et = torch.dot(w_ext, sum_et_per_ext)
    
    term_cross = (sum_it + sum_et) / (total_weight_source * n_t)
    
    # --- Term 2: E[d(Source, Source)] ---
    # Split into Int-Int, Int-Ext, Ext-Ext
    
    # 2a. Internal <-> Internal
    d_ii = torch.cdist(Xi, Xi)
    sum_ii = d_ii.sum()
    
    # 2b. Internal <-> External
    d_ie = torch.cdist(Xi, Xe) # (n_i, n_e)
    # Sum over int (dim 0), then dot with weights (dim 1)
    sum_ie_per_ext = d_ie.sum(dim=0) # (n_e,)
    sum_ie = torch.dot(w_ext, sum_ie_per_ext)
    
    # 2c. External <-> External (Weighted both sides)
    d_ee = torch.cdist(Xe, Xe) # (n_e, n_e)
    # w^T * D * w
    sum_ee = torch.dot(w_ext, torch.mv(d_ee, w_ext))
    
    term_self_source = (sum_ii + 2 * sum_ie + sum_ee) / (total_weight_source**2)
    
    # --- Term 3: E[d(Target, Target)] ---
    d_tt = torch.cdist(Xt, Xt)
    sum_tt = d_tt.sum()
    term_self_target = sum_tt / (n_t**2)
    
    energy_dist = 2 * term_cross - term_self_source - term_self_target
    
    return energy_dist.item()
