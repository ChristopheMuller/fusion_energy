import numpy as np
import torch

def generate_covariates(n, dim, mean, var=1.0):
    cov = np.eye(dim) * var
    return np.random.multivariate_normal(mean, cov, n)

def generate_outcomes_nonlinear(X, treatment_effect):
    # Y depends on X0 (quadratic) and interaction X1*X2
    # X3, X4... are linear
    # X5, X6... are noise ("toenails")
    y0 = 2.0 * (X[:, 0] - 1)**2 + 1.5 * np.sin(np.pi * X[:, 1] * X[:, 2]) + 0.3 * X[:, 3] - 0.2 * X[:, 4] + np.random.normal(0, 0.5, X.shape[0])
    y1 = y0 + treatment_effect
    return y0, y1

def create_complex_dataset(n_treat, n_control_rct, n_external, dim, rct_bias=0.5, ext_bias=1.):
    # Target (RCT Treat): Centered at 1.0
    mean_target = np.ones(dim)
    X_target = generate_covariates(n_treat, dim, mean_target, var=1.0)
    
    # Internal Control: Biased (e.g. dropouts), Centered at 1.0 - bias
    mean_control = mean_target - rct_bias
    X_control = generate_covariates(n_control_rct, dim, mean_control, var=1.0)
    
    # External Control: Heavily biased, Centered at 1.0 - large_bias
    # But wider variance, so it overlaps
    mean_ext = mean_target - ext_bias
    X_ext = generate_covariates(n_external, dim, mean_ext, var=2.0)
    
    # Outcomes
    tau = 2.5
    _, Y_target = generate_outcomes_nonlinear(X_target, tau)
    Y0_control, _ = generate_outcomes_nonlinear(X_control, tau)
    Y0_ext, _ = generate_outcomes_nonlinear(X_ext, tau)
    
    return {
        "target": {"X": X_target, "Y": Y_target},
        "internal": {"X": X_control, "Y": Y0_control},
        "external": {"X": X_ext, "Y": Y0_ext},
        "tau": tau
    }