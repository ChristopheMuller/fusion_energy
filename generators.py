import numpy as np
from structures import PotentialOutcomes

class DataGenerator:
    def __init__(self, dim: int = 5):
        self.dim = dim

    def _generate_covariates(self, n, mean, var):
        cov = np.eye(self.dim) * var
        return np.random.multivariate_normal(mean, cov, n)

    def _generate_outcomes(self, X, beta=None, treatment_effect=None):
        if beta is None:
            beta = np.ones(self.dim)
        if treatment_effect is None:
            treatment_effect = 0.
        n = X.shape[0]
        y0 = X @ beta + np.random.normal(0, 0.5, n)
        y1 = y0 + treatment_effect
        return y0, y1

    def generate_rct_pool(self, n, mean, var, effect_size=1.0):
        X = self._generate_covariates(n, mean, var)
        beta = np.random.uniform(-1, 1, self.dim)
        Y0, Y1 = self._generate_outcomes(X, beta, effect_size)
        return PotentialOutcomes(X=X, Y0=Y0, Y1=Y1)

    def generate_external_pool(self, n, mean, var, beta_bias=0.0):
        X = self._generate_covariates(n, mean, var)
        beta = np.random.uniform(-1, 1, self.dim) + beta_bias
        Y0, _ = self._generate_outcomes(X, beta, 0) 
        return PotentialOutcomes(X=X, Y0=Y0, Y1=None)