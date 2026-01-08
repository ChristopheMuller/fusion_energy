import numpy as np
from scipy.spatial.distance import cdist

def compute_distance_matrix(X1, X2):
    return cdist(X1, X2, metric='euclidean')

def calculate_att_error(estimated_att, true_att):
    return estimated_att - true_att

def calculate_rmse(errors):
    return np.sqrt(np.mean(np.array(errors)**2))

def calculate_ess(weights):
    weights = np.array(weights)
    return (np.sum(weights)**2) / np.sum(weights**2)