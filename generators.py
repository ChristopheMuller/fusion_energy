import numpy as np
from structures import PotentialOutcomes

class DataGenerator:
    def __init__(self, dim: int = 5, beta = None):
        self.dim = dim
        if beta is None:
            self.beta = np.random.uniform(1, 3, dim)
        else:
            self.beta = beta
    def _generate_covariates(self, n, mean, var, corr=0.0):
        # Toeplitz structure
        idx = np.arange(self.dim)
        cov = var * (corr ** np.abs(idx[:, None] - idx[None, :]))
        return np.random.multivariate_normal(mean, cov, n)

    def _generate_outcomes(self, X, treatment_effect, beta_bias=None):
        """
        Generates potential outcomes. 
        beta_bias: Optional vector to add to the base beta (simulating concept drift).
        """
        n = X.shape[0]
        
        if callable(treatment_effect):
            treatment_effect = treatment_effect(X)
        elif np.isscalar(treatment_effect):
            treatment_effect = np.ones(n) * treatment_effect
        else:
            raise ValueError("treatment_effect must be a scalar or a callable function.")

        current_beta = self.beta if beta_bias is None else self.beta + beta_bias
        y0 = X @ current_beta + np.random.normal(0, 0.5, n)
        y1 = y0 + treatment_effect
        return y0, y1

    def generate_rct_pool(self, n, mean, var, corr=0.0, treatment_effect=None):
        X = self._generate_covariates(n, mean, var, corr=corr)
        Y0, Y1 = self._generate_outcomes(X, treatment_effect=treatment_effect, beta_bias=None)
        return PotentialOutcomes(X=X, Y0=Y0, Y1=Y1)

    def generate_external_pool(self, n, mean, var, corr=0.0, beta_bias=0.0):
        X = self._generate_covariates(n, mean, var, corr=corr)
        bias_vec = np.ones(self.dim) * beta_bias
        Y0, _ = self._generate_outcomes(X, treatment_effect=0, beta_bias=bias_vec)
        return PotentialOutcomes(X=X, Y0=Y0, Y1=None)