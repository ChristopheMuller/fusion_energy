import pytest
import inspect
import numpy as np
import estimator
from estimator.base import BaseEstimator
from structures import EstimationResult

def get_all_estimators():
    estimators = []
    for name, obj in inspect.getmembers(estimator):
        if inspect.isclass(obj) and issubclass(obj, BaseEstimator) and obj is not BaseEstimator:
            # Skip REstimator as requested
            if name == "REstimator":
                continue
            estimators.append(obj)
    return estimators

@pytest.mark.parametrize("estimator_cls", get_all_estimators())
def test_estimator_basic(estimator_cls, split_data):
    # Instantiate with small iterations for speed
    init_params = inspect.signature(estimator_cls.__init__).parameters
    kwargs = {}
    if "n_iter" in init_params:
        kwargs["n_iter"] = 10
    if "max_iter" in init_params:
        kwargs["max_iter"] = 10
    if "n_iter_weights" in init_params:
        kwargs["n_iter_weights"] = 50
    if "k_best" in init_params:
        kwargs["k_best"] = 2

    # For Optimal estimators, set max_external small
    if "max_external" in init_params:
        kwargs["max_external"] = 5
        kwargs["step"] = 5

    est = estimator_cls(**kwargs)
    res = est.estimate(split_data)

    assert isinstance(res, EstimationResult)
    assert isinstance(res.ate_est, (float, np.float32, np.float64))

    # Check weights dimensions
    n_ext = len(split_data.X_external)
    assert len(res.weights_external) == n_ext
    assert len(res.weights_continuous) == n_ext

    # Check that ATE is not NaN
    assert not np.isnan(res.ate_est)
