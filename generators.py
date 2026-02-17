import numpy as np
from structures import PotentialOutcomes

class DataGenerator:
    def __init__(self, dim: int = 5, beta=None, non_linear_covariates: bool = False, non_linear_outcome: bool = False):
        self.dim = dim
        self.non_linear_covariates = non_linear_covariates
        self.non_linear_outcome = non_linear_outcome
        if beta is None:
            self.beta = np.random.uniform(1, 3, dim)
        else:
            self.beta = beta

    def _generate_covariates(self, n, mean, var, corr=0.0):
        idx = np.arange(self.dim)
        cov = var * (corr ** np.abs(idx[:, None] - idx[None, :]))
        X_raw = np.random.multivariate_normal(np.zeros(self.dim), cov, n).astype(np.float32)

        if self.non_linear_covariates:
            X_trans = np.zeros_like(X_raw)
            for d in range(self.dim):
                if d % 3 == 0:
                    X_trans[:, d] = np.exp(X_raw[:, d] / 2)
                elif d % 3 == 1:
                    X_trans[:, d] = X_raw[:, d] ** 2
                else:
                    X_trans[:, d] = np.sin(X_raw[:, d])
            
            X_trans = (X_trans - np.mean(X_trans, axis=0)) / np.std(X_trans, axis=0)
            X = X_trans * np.sqrt(var)
        else:
            X = X_raw

        X = X + mean
        return X.astype(np.float32)

    def _generate_outcomes(self, X, treatment_effect, beta_bias=None):
        n = X.shape[0]
        
        if callable(treatment_effect):
            treatment_effect = treatment_effect(X)
        elif np.isscalar(treatment_effect):
            treatment_effect = np.ones(n) * treatment_effect
        else:
            raise ValueError("treatment_effect must be a scalar or a callable function.")

        current_beta = self.beta if beta_bias is None else self.beta + beta_bias
        
        linear_signal = X @ current_beta

        if self.non_linear_outcome:
            raw_signal = np.zeros(n)
            for d in range(self.dim):
                if d % 4 == 0:
                    raw_signal += 3 * np.sin(X[:, d] * current_beta[d])
                elif d % 4 == 1:
                    raw_signal += 2 * np.log(np.abs(X[:, d]) + 1) * current_beta[d]
                elif d % 4 == 2:
                    raw_signal += 0.5 * (X[:, d] ** 2) * current_beta[d]
                else:
                    raw_signal += 1.5 * X[:, d] * current_beta[d]
            
            raw_signal = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)
            signal = raw_signal * np.std(linear_signal) + np.mean(linear_signal)
        else:
            signal = linear_signal

        y0 = (signal + np.random.normal(0, 0.5, n)).astype(np.float32)
        y1 = (y0 + treatment_effect).astype(np.float32)
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