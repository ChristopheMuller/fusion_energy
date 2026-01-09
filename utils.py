import numpy as np
from scipy.spatial.distance import cdist

def compute_energy_distance_numpy(X1, X2):
    # Standard unweighted Energy Distance
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    d12 = cdist(X1, X2, 'euclidean').mean()
    d11 = cdist(X1, X1, 'euclidean').mean()
    d22 = cdist(X2, X2, 'euclidean').mean()
    
    return 2 * d12 - d11 - d22

def calculate_bias_rmse(estimates, true_val):
    est = np.array(estimates)
    bias = np.mean(est) - true_val
    rmse = np.sqrt(np.mean((est - true_val)**2))
    return bias, rmse