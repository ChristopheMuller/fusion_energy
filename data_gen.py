import numpy as np

def generate_covariates(n, dim, mean, var):
    cov = np.eye(dim) * var
    return np.random.multivariate_normal(mean, cov, n)

def generate_outcomes(X, treatment_effect, beta=None):
    # Y depends on X linearly
    n, d = X.shape
    
    if beta is None:
        beta = np.random.uniform(-2, 2, size=d)
    y0 = X @ beta + np.random.normal(0, 0.5, n)
    
    if callable(treatment_effect):
        effect = treatment_effect(X)
    else:
        effect = treatment_effect
        
    y1 = y0 + effect
    return y0, y1

def create_complex_dataset(n_treat, n_control_rct, n_external, dim, tau, rct_bias=0., ext_bias=1., rct_var=1.0, ext_var=2.0):
    
    # Target (RCT Treat): Centered at 1.0
    mean_target = np.ones(dim)
    X_target = generate_covariates(n_treat, dim, mean_target, var=rct_var)

    # Internal Control: potentially biased (e.g. dropouts), Centered at 1.0 - bias
    mean_control = mean_target - rct_bias
    X_control = generate_covariates(n_control_rct, dim, mean_control, var=rct_var)

    mean_ext = mean_target - ext_bias
    X_ext = generate_covariates(n_external, dim, mean_ext, var=ext_var)

    # Outcomes
    beta = np.ones(dim)
    Y0_target, Y1_target = generate_outcomes(X_target, tau, beta=beta)
    Y0_control, Y1_control = generate_outcomes(X_control, tau, beta=beta)
    Y0_ext, Y1_ext = generate_outcomes(X_ext, tau, beta=beta)

    true_att = np.mean(Y1_target - Y0_target)

    return {
        "target": {"X": X_target, "Y0": Y0_target, "Y1": Y1_target, "Y": Y1_target},
        "internal": {"X": X_control, "Y0": Y0_control, "Y1": Y1_control, "Y": Y0_control},
        "external": {"X": X_ext, "Y0": Y0_ext, "Y1": Y1_ext, "Y": Y0_ext},
        "true_att": true_att
    }