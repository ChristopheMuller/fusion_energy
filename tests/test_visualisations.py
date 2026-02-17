import os
import pytest
import numpy as np
from visualisations import plot_error_boxplots, plot_pca_weights, plot_metric_curves, plot_mse_decomposition, plot_energy_distance, plot_weight_ranks
from structures import EstimationResult

def test_plot_error_boxplots(tmp_path):
    class MockResult:
        def __init__(self, bias):
            self.bias = bias

    class MockLog:
        def __init__(self, biases):
            self.results = [MockResult(b) for b in biases]

    logs = {
        "Method1": MockLog([0.1, 0.2, -0.1]),
        "Method2": MockLog([0.5, 0.4, 0.6])
    }

    plot_file = tmp_path / "test_boxplot.png"
    plot_error_boxplots(logs, filename=str(plot_file))

    assert plot_file.exists()
    assert plot_file.stat().st_size > 0

def test_plot_pca_weights(split_data, tmp_path):
    # Create a dummy EstimationResult
    n_ext = len(split_data.X_external)
    est_result = EstimationResult(
        ate_est=1.9,
        bias=-0.1,
        weights_continuous=np.random.rand(n_ext),
        weights_external=np.random.rand(n_ext)
    )

    plot_filename = tmp_path / "pca_weights.png"
    # plot_pca_weights creates multiple files with suffixes _continuous.png and _external.png
    plot_pca_weights(split_data, est_result, "Test Title", str(plot_filename))

    base_filename, ext = os.path.splitext(str(plot_filename))
    continuous_plot = f"{base_filename}_continuous{ext}"
    external_plot = f"{base_filename}_external{ext}"

    assert os.path.exists(continuous_plot)
    assert os.path.getsize(continuous_plot) > 0
    assert os.path.exists(external_plot)
    assert os.path.getsize(external_plot) > 0

def test_plot_metric_curves(tmp_path):
    class MockResult:
        def __init__(self, bias, ate_est, energy_distance, weights_external):
            self.bias = bias
            self.ate_est = ate_est
            self.energy_distance = energy_distance
            self.weights_external = weights_external
        def sum_of_weights_external(self):
            return np.sum(self.weights_external)
        def n_external_used(self):
            return len(self.weights_external)

    class MockLog:
        def __init__(self, results):
            self.results = results

    res1 = MockResult(0.1, 1.1, 0.5, np.array([1, 1]))
    res2 = MockResult(0.05, 1.05, 0.3, np.array([1, 1, 1]))
    
    logs = {
        "Method1": MockLog([res1, res1]),
        "Method2": MockLog([res2, res2])
    }

    plot_file = tmp_path / "metric_curves.png"
    plot_metric_curves(logs, filename=str(plot_file))

    assert plot_file.exists()
    assert plot_file.stat().st_size > 0
