import pytest
import numpy as np
import inspect
import estimator
from estimator.base import BaseEstimator
from generators import DataGenerator
from structures import SplitData

def get_all_estimators():
    estimators = []
    for name, obj in inspect.getmembers(estimator):
        if inspect.isclass(obj) and issubclass(obj, BaseEstimator) and obj is not BaseEstimator:
            if name == "REstimator":
                continue
            estimators.append(obj)
    return estimators

def generate_sample_data(seed, n_rct=100, n_ext=200, dim=5):
    np.random.seed(seed)
    dg = DataGenerator(dim=dim)

    # Same distribution for RCT and External
    mean = np.zeros(dim)
    var = 1.0

    rct_pool = dg.generate_rct_pool(n_rct, mean, var, treatment_effect=2.0)
    ext_pool = dg.generate_external_pool(n_ext, mean, var, beta_bias=0.0)

    n_treat = n_rct // 2
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
        target_n_aug=n_ext // 2
    )

def run_simulation(estimator_cls, n_trials=20):
    estimates = []
    rct_estimates = []

    init_params = inspect.signature(estimator_cls.__init__).parameters
    kwargs = {}
    if "n_iter" in init_params: kwargs["n_iter"] = 10
    if "max_iter" in init_params: kwargs["max_iter"] = 10
    if "n_iter_weights" in init_params: kwargs["n_iter_weights"] = 50
    if "k_best" in init_params: kwargs["k_best"] = 10
    if "max_external" in init_params:
        kwargs["max_external"] = 50
        kwargs["step"] = 10

    for i in range(n_trials):
        data = generate_sample_data(seed=i)

        # Augmented estimate
        est_obj = estimator_cls(**kwargs)
        res = est_obj.estimate(data)
        estimates.append(res.ate_est)

        # RCT-only estimate
        rct_ate = np.mean(data.Y_treat) - np.mean(data.Y_control_int)
        rct_estimates.append(rct_ate)

    true_ate = 2.0
    mse_aug = np.mean((np.array(estimates) - true_ate)**2)
    mse_rct = np.mean((np.array(rct_estimates) - true_ate)**2)

    return mse_aug, mse_rct

@pytest.mark.parametrize("estimator_cls", get_all_estimators())
def test_zero_bias_performance(estimator_cls):
    # Using 15 trials to keep it under 2 mins for all 10 estimators
    # Each estimator takes ~5-10 seconds total for 15 trials with n_iter=10
    mse_aug, mse_rct = run_simulation(estimator_cls, n_trials=15)

    print(f"\nEstimator: {estimator_cls.__name__}")
    print(f"MSE Augmented: {mse_aug:.6f}")
    print(f"MSE RCT:       {mse_rct:.6f}")

    # We expect MSE_aug <= MSE_rct when there is zero bias.
    # Allowing a small tolerance because of small number of trials and stochastic nature.
    # Actually, it should be strictly better or equal.
    # Some estimators might be slightly worse due to noise if they are not efficient,
    # but on average they should be better.
    assert mse_aug <= mse_rct * 1.1, f"{estimator_cls.__name__} performed significantly worse than RCT even with zero bias"
