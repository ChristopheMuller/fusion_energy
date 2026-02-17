import pytest
import os
import re
import numpy as np
from estimator.r_bridge import REstimator
from structures import EstimationResult

def get_r_estimators():
    r_folder = "R"
    estimators = []
    if not os.path.exists(r_folder):
        return estimators

    for filename in os.listdir(r_folder):
        if filename.endswith(".R"):
            filepath = os.path.join(r_folder, filename)
            with open(filepath, "r") as f:
                content = f.read()
                # Find all functions starting with estimate_
                # Pattern matches estimate_something followed by <- function or = function
                matches = re.findall(r"(\bestimate_[a-zA-Z0-9_]+)\s*(?:<-|=)\s*function", content)
                for func_name in matches:
                    estimators.append((filepath, func_name))
    return estimators

@pytest.mark.parametrize("r_script_path, r_func_name", get_r_estimators())
def test_r_estimator_basic(r_script_path, r_func_name, split_data):
    est = REstimator(r_script_path=r_script_path, r_func_name=r_func_name)
    res = est.estimate(split_data)

    assert isinstance(res, EstimationResult)
    assert isinstance(res.ate_est, (float, np.float32, np.float64))

    # Check weights dimensions
    n_ext = len(split_data.X_external)
    assert len(res.weights_external) == n_ext
    assert len(res.weights_continuous) == n_ext

    # Check that ATE is not NaN
    assert not np.isnan(res.ate_est), f"ATE estimate is NaN for {r_func_name} in {r_script_path}"
