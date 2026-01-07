import unittest
import numpy as np
from data_gen import create_dataset
from methods import optimize_weights, refined_sampling, inverse_propensity_weighting
from utils import compute_distance_matrix, calculate_att_error

class TestSimulation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_rct_treat = 20
        self.n_rct_control = 10
        self.n_ext = 50
        self.dim = 3
        self.shift_ext = 1.0
        self.beta = np.ones(self.dim)
        self.tau = 2.0
        self.data = create_dataset(
            self.n_rct_treat, self.n_rct_control, self.n_ext, 
            self.dim, self.shift_ext, self.beta, self.tau
        )

    def test_create_dataset_structure(self):
        """Test if the dataset has the correct structure and shapes."""
        self.assertIn("rct_treat", self.data)
        self.assertIn("rct_control", self.data)
        self.assertIn("external", self.data)
        
        self.assertEqual(self.data["rct_treat"]["X"].shape, (self.n_rct_treat, self.dim))
        self.assertEqual(self.data["rct_control"]["X"].shape, (self.n_rct_control, self.dim))
        self.assertEqual(self.data["external"]["X"].shape, (self.n_ext, self.dim))
        
        # Check treatment assignment labels
        self.assertTrue(np.all(self.data["rct_treat"]["A"] == 1))
        self.assertTrue(np.all(self.data["rct_control"]["A"] == 0))
        self.assertTrue(np.all(self.data["external"]["A"] == 0))

    def test_compute_distance_matrix(self):
        """Test distance matrix calculation."""
        X1 = np.array([[0, 0], [1, 1]])
        X2 = np.array([[0, 0], [1, 0]])
        # Distances: 
        # (0,0)-(0,0)=0, (0,0)-(1,0)=1
        # (1,1)-(0,0)=sqrt(2), (1,1)-(1,0)=1
        
        D = compute_distance_matrix(X1, X2)
        expected = np.array([[0.0, 1.0], [np.sqrt(2), 1.0]])
        np.testing.assert_allclose(D, expected)

    def test_optimize_weights(self):
        """Test if optimized weights sum to 1 and are non-negative."""
        X_source = self.data["external"]["X"][:20] # Use subset for speed
        X_target = self.data["rct_treat"]["X"][:10]
        
        weights = optimize_weights(X_source, X_target)
        
        # Check sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        
        # Check non-negativity (allow small numerical errors)
        self.assertTrue(np.all(weights >= -1e-9))

    def test_refined_sampling(self):
        """Test refined sampling returns correct number of samples."""
        X_pool = self.data["external"]["X"]
        Y_pool = self.data["external"]["Y"]
        X_target = self.data["rct_treat"]["X"]
        
        # Create dummy weights
        weights = np.ones(len(X_pool)) / len(X_pool)
        
        n_select = 15
        X_sample, Y_sample = refined_sampling(
            X_pool, Y_pool, weights, X_target, n_select=n_select, K=10
        )
        
        self.assertEqual(X_sample.shape[0], n_select)
        self.assertEqual(X_sample.shape[1], self.dim)
        self.assertEqual(Y_sample.shape[0], n_select)

    def test_calculate_att_error(self):
        """Test ATT error calculation."""
        est = 5.0
        true = 2.0
        self.assertEqual(calculate_att_error(est, true), 3.0)

    def test_non_linear_shift(self):
        """Test that quadratic shift produces different external data."""
        np.random.seed(999)
        data_linear = create_dataset(
            10, 10, 10, self.dim, self.shift_ext, self.beta, self.tau, shift_type="linear"
        )
        
        np.random.seed(999)
        data_quad = create_dataset(
            10, 10, 10, self.dim, self.shift_ext, self.beta, self.tau, shift_type="quadratic"
        )
        
        # External data should be different
        self.assertFalse(np.allclose(data_linear["external"]["X"], data_quad["external"]["X"]))
        
        # RCT data should be identical (controlled by same seed, no shift applied)
        self.assertTrue(np.allclose(data_linear["rct_treat"]["X"], data_quad["rct_treat"]["X"]))

    def test_inverse_propensity_weighting(self):
        """Test if IPW weights sum to 1 and are non-negative."""
        X_source = self.data["external"]["X"][:50]
        X_target = self.data["rct_treat"]["X"][:20]
        
        weights = inverse_propensity_weighting(X_source, X_target)
        
        # Check sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        
        # Check non-negativity
        self.assertTrue(np.all(weights >= 0))
        
        # Check dimensions
        self.assertEqual(len(weights), len(X_source))

if __name__ == '__main__':
    unittest.main()
