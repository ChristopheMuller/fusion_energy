import pytest
import numpy as np
from structures import EstimationResult
from estimator import (
    IPWEstimator,
    EnergyMatchingEstimator,
    EnergyWeightingEstimator,
    DummyMatchingEstimator,
    OptimalEnergyMatchingEstimator
)

def verify_estimation_result(res: EstimationResult, n_int: int, n_ext: int):
    assert isinstance(res, EstimationResult)
    assert isinstance(res.ate_est, float)
    # The current implementation seems to return weights of size n_ext for both internal and external
    assert len(res.weights_continuous) == n_ext
    assert len(res.weights_external) == n_ext

def test_ipw_estimator(split_data):
    est = IPWEstimator(cv=2, max_iter=50)
    res = est.estimate(split_data)
    verify_estimation_result(res, len(split_data.X_control_int), len(split_data.X_external))

def test_energy_matching_estimator(split_data):
    est = EnergyMatchingEstimator(n_iter=10, k_best=5)
    res = est.estimate(split_data)
    verify_estimation_result(res, len(split_data.X_control_int), len(split_data.X_external))
    assert res.energy_distance is not None

def test_optimal_energy_matching_estimator(split_data):
    # Set max_external small to be faster
    n_ext_total = split_data.X_external.shape[0]
    est = OptimalEnergyMatchingEstimator(max_external=min(10, n_ext_total), step=2, n_iter=10, k_best=5)
    res = est.estimate(split_data)
    verify_estimation_result(res, len(split_data.X_control_int), len(split_data.X_external))
    assert res.energy_distance is not None
    # Check that n_external_used is one of the candidates (0, 2, 4, ..., or max)
    # The actual used might be 0 if that's optimal
    assert res.n_external_used() in range(0, min(10, n_ext_total) + 1)

def test_energy_weighting_estimator(split_data):
    est = EnergyWeightingEstimator(n_iter=10)
    res = est.estimate(split_data)
    verify_estimation_result(res, len(split_data.X_control_int), len(split_data.X_external))

def test_dummy_matching_estimator(split_data):
    est = DummyMatchingEstimator()
    res = est.estimate(split_data)
    verify_estimation_result(res, len(split_data.X_control_int), len(split_data.X_external))

def test_dummy_matching_estimator_zero_aug(split_data):
    est = DummyMatchingEstimator()
    res = est.estimate(split_data, n_external=0)
    assert res.n_external_used() == 0
    verify_estimation_result(res, len(split_data.X_control_int), len(split_data.X_external))
