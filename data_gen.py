import numpy as np

def generate_covariates(n_samples, dim, mean_shift=0.0, shift_type="linear"):
    mean = np.zeros(dim) + mean_shift
    cov = np.eye(dim)
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    if shift_type == "quadratic":
        X = X + 0.5 * (X**2 - 1)
        
    return X

def generate_potential_outcomes(X, beta, treatment_effect):
    n = X.shape[0]
    noise = np.random.normal(0, 1, n)
    Y0 = X @ beta + noise
    Y1 = Y0 + treatment_effect
    return Y0, Y1

def create_dataset(n_rct_treat, n_rct_control, n_ext, dim, shift_ext, beta, tau, shift_type="linear"):
    X_rct_1 = generate_covariates(n_rct_treat, dim, mean_shift=0.0)
    _, Y1_rct_1 = generate_potential_outcomes(X_rct_1, beta, tau)
    
    X_rct_0 = generate_covariates(n_rct_control, dim, mean_shift=0.0)
    Y0_rct_0, _ = generate_potential_outcomes(X_rct_0, beta, tau)
    
    X_ext = generate_covariates(n_ext, dim, mean_shift=shift_ext, shift_type=shift_type)
    Y0_ext, _ = generate_potential_outcomes(X_ext, beta, tau)
    
    data = {
        "rct_treat": {"X": X_rct_1, "Y": Y1_rct_1, "A": 1},
        "rct_control": {"X": X_rct_0, "Y": Y0_rct_0, "A": 0},
        "external": {"X": X_ext, "Y": Y0_ext, "A": 0}
    }
    
    return data