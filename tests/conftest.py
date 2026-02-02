import pytest
import numpy as np
from structures import PotentialOutcomes, SplitData

@pytest.fixture
def rct_pool():
    np.random.seed(42)
    n = 20
    dim = 2
    X = np.random.normal(0, 1, (n, dim))
    Y0 = X @ np.array([1, 2]) + np.random.normal(0, 0.1, n)
    Y1 = Y0 + 2.0
    return PotentialOutcomes(X=X, Y0=Y0, Y1=Y1)

@pytest.fixture
def ext_pool():
    np.random.seed(43)
    n = 30
    dim = 2
    X = np.random.normal(0.5, 1, (n, dim))
    Y0 = X @ np.array([1.1, 1.9]) + np.random.normal(0, 0.1, n)
    return PotentialOutcomes(X=X, Y0=Y0, Y1=None)

@pytest.fixture
def split_data(rct_pool, ext_pool):
    # Manually create a SplitData object for testing estimators
    n_treat = 10
    n_control = 10
    X_treat = rct_pool.X[:n_treat]
    Y_treat = rct_pool.Y1[:n_treat]
    X_control_int = rct_pool.X[n_treat:]
    Y_control_int = rct_pool.Y0[n_treat:]

    return SplitData(
        X_treat=X_treat,
        Y_treat=Y_treat,
        X_control_int=X_control_int,
        Y_control_int=Y_control_int,
        X_external=ext_pool.X,
        Y_external=ext_pool.Y0,
        true_sate=2.0,
        target_n_aug=5
    )
