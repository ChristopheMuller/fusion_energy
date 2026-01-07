import numpy as np
from scipy.optimize import minimize
from utils import compute_distance_matrix

def energy_loss(w, D_xs, D_xx):
    term1 = 2 * np.dot(w, np.mean(D_xs, axis=1))
    term2 = np.dot(w, np.dot(D_xx, w))
    return term1 - term2

def optimize_weights(X_source, X_target):
    n_source = X_source.shape[0]
    
    D_xs = compute_distance_matrix(X_source, X_target)
    D_xx = compute_distance_matrix(X_source, X_source)
    
    w0 = np.ones(n_source) / n_source
    
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    bounds = [(0, 1) for _ in range(n_source)]
    
    result = minimize(
        energy_loss,
        w0,
        args=(D_xs, D_xx),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-6, 'disp': False}
    )
    
    return result.x

def energy_distance_unweighted(X_sample, X_target):
    n = X_sample.shape[0]
    m = X_target.shape[0]
    
    D_xs = compute_distance_matrix(X_sample, X_target)
    D_xx = compute_distance_matrix(X_sample, X_sample)
    
    term1 = 2 * np.mean(D_xs)
    term2 = np.mean(D_xx) 
    
    return term1 - term2

def refined_sampling(X_pool, Y_pool, w_opt, X_target, n_select, K=100):
    best_dist = np.inf
    best_indices = None
    
    pool_indices = np.arange(len(X_pool))
    prob_dist = w_opt / np.sum(w_opt)
    
    for k in range(K):
        candidate_indices = np.random.choice(
            pool_indices, size=n_select, replace=True, p=prob_dist
        )
        X_cand = X_pool[candidate_indices]
        
        dist = energy_distance_unweighted(X_cand, X_target)
        
        if dist < best_dist:
            best_dist = dist
            best_indices = candidate_indices
            
    return X_pool[best_indices], Y_pool[best_indices]