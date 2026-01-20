import numpy as np
from structures import PotentialOutcomes

class DataGenerator:
    def __init__(self, dim: int = 5, beta = None):
        self.dim = dim
        if beta is None:
            self.beta = np.random.uniform(1, 3, dim)
        else:
            self.beta = beta
    def _generate_covariates(self, n, mean, var):
        cov = np.eye(self.dim) * var
        return np.random.multivariate_normal(mean, cov, n)

    def _generate_outcomes(self, X, treatment_effect=0., beta_bias=None):
        """
        Generates potential outcomes. 
        beta_bias: Optional vector to add to the base beta (simulating concept drift).
        """
        n = X.shape[0]
        
        current_beta = self.beta if beta_bias is None else self.beta + beta_bias
        y0 = X @ current_beta + np.random.normal(0, 0.5, n)
        y1 = y0 + treatment_effect
        return y0, y1

    def generate_rct_pool(self, n, mean, var, effect_size=1.0):
        X = self._generate_covariates(n, mean, var)
        Y0, Y1 = self._generate_outcomes(X, treatment_effect=effect_size, beta_bias=None)
        return PotentialOutcomes(X=X, Y0=Y0, Y1=Y1)

    def generate_external_pool(self, n, mean, var, beta_bias=0.0):
        X = self._generate_covariates(n, mean, var)
        bias_vec = np.ones(self.dim) * beta_bias
        Y0, _ = self._generate_outcomes(X, treatment_effect=0, beta_bias=bias_vec)
        return PotentialOutcomes(X=X, Y0=Y0, Y1=None)