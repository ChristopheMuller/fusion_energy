import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import compute_distance_matrix

def _project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def optimize_weights(X_source, X_target, max_iter=1000, tol=1e-6):
    n_source = X_source.shape[0]
    
    D_xs = compute_distance_matrix(X_source, X_target)
    D_xx = compute_distance_matrix(X_source, X_source)
    
    mean_D_xs = np.mean(D_xs, axis=1)
    L = 2 * np.max(np.sum(np.abs(D_xx), axis=1))
    lr = 1.0 / L
    
    w = np.ones(n_source) / n_source
    
    for _ in range(max_iter):
        grad = 2 * mean_D_xs - 2 * D_xx.dot(w)
        w_new = _project_simplex(w - lr * grad)
        
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
            
    return w

def inverse_propensity_weighting(X_source, X_target):
    y_source = np.zeros(X_source.shape[0])
    y_target = np.ones(X_target.shape[0])
    
    X_combined = np.vstack([X_source, X_target])
    y_combined = np.concatenate([y_source, y_target])
    
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_combined, y_combined)
    
    propensity_scores = clf.predict_proba(X_source)[:, 1]
    
    propensity_scores = np.clip(propensity_scores, 0.001, 0.999)
    
    weights = propensity_scores / (1 - propensity_scores)
    
    weights = weights / np.sum(weights)
    
    return weights

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
