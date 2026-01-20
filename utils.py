import numpy as np
from scipy.spatial.distance import cdist

def compute_energy_distance_numpy(X1, X2):
    # Standard unweighted Energy Distance
    if X1 is None:
        return None, None, None, None

    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    d12 = cdist(X1, X2, 'euclidean').mean()
    d11 = cdist(X1, X1, 'euclidean').mean()
    d22 = cdist(X2, X2, 'euclidean').mean()
    
    return 2 * d12 - d11 - d22, d12, d11, d22

def compute_weighted_energy_distance(X_source, X_target, weights=None):
    """
    Computes Energy Distance between a Weighted Source (X_source, weights) 
    and an Unweighted Target (X_target).
    
    Energy = 2 * E[d(S, T)] - E[d(S, S)] - E[d(T, T)]
    """
    if X_source is None or X_target is None:
        return None, None, None, None
        
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    
    # Default to uniform weights if None
    if weights is None:
        weights = np.ones(n_source) / n_source
    else:
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
    # 1. Source-Target Term (Weighted)
    # d_st matrix: (n_source, n_target)
    # We want sum_{i,j} w_i * (1/n_t) * d(x_i, y_j)
    # = (1/n_t) * sum_i w_i * (sum_j d(x_i, y_j))
    d_st = cdist(X_source, X_target, 'euclidean')
    # Mean distance from each source point to all target points
    d_st_mean_per_source = d_st.mean(axis=1) # Shape (n_source,)
    term_st = np.dot(weights, d_st_mean_per_source)
    
    # 2. Source-Source Term (Weighted)
    # sum_{i,j} w_i * w_j * d(x_i, x_j)
    d_ss = cdist(X_source, X_source, 'euclidean')
    # w^T * D_ss * w
    term_ss = np.dot(weights, np.dot(d_ss, weights))
    
    # 3. Target-Target Term (Unweighted)
    d_tt = cdist(X_target, X_target, 'euclidean')
    term_tt = d_tt.mean()
    
    energy = 2 * term_st - term_ss - term_tt
    
    return energy, term_st, term_ss, term_tt


def calculate_bias_rmse(estimates, true_val):
    est = np.array(estimates)
    bias = np.mean(est) - true_val
    rmse = np.sqrt(np.mean((est - true_val)**2))
    return bias, rmse