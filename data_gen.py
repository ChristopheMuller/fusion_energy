import numpy as np

def generate_covariates(n, dim, mean, var):
    cov = np.eye(dim) * var
    return np.random.multivariate_normal(mean, cov, n)

def generate_outcomes_nonlinear(X, treatment_effect, beta=None):
    # Y depends on X linearly
    n, d = X.shape
    
    if beta is None:
        beta = np.random.uniform(-2, 2, size=d)
    y0 = X @ beta + np.random.normal(0, 0.5, n)
    y1 = y0 + treatment_effect

    # y0 = np.zeros(n)
    # if d >= 1:
    #     y0 += np.sin(X[:, 0])
    # if d >= 2:
    #     y0 += np.log(np.abs(X[:, 1]) + 1)
    # if d >= 3:
    #     y0 += X[:, 2] ** 2
    # if d >= 4:
    #     y0 += np.exp(-X[:, 3] ** 2 / 2)
    # if d >= 6:
    #     y0 += np.ones(d-4) @ X[:, 4:d].T

    # y1 = y0 + treatment_effect

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
    Y0_target, Y_target = generate_outcomes_nonlinear(X_target, tau, beta=beta)
    Y0_control, _ = generate_outcomes_nonlinear(X_control, tau, beta=beta)
    Y0_ext, _ = generate_outcomes_nonlinear(X_ext, tau, beta=beta)

    true_att = np.mean(Y_target - Y0_target)

    return {
        "target": {"X": X_target, "Y": Y_target},
        "internal": {"X": X_control, "Y": Y0_control},
        "external": {"X": X_ext, "Y": Y0_ext},
        "tau": tau,
        "true_att": true_att
    }