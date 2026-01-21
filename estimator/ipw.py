import numpy as np
from sklearn.linear_model import LogisticRegression
from structures import SplitData, EstimationResult
from .base import BaseEstimator

class IPWEstimator(BaseEstimator):
    """
    Estimates ATT using Inverse Probability Weighting (IPW) to reweight 
    external controls to look like the internal/target population.
    """
    def estimate(self, data: SplitData) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        
        y1_mean = np.mean(data.Y_treat)

        # Edge case: No external data or augmentation not requested
        if data.target_n_aug == 0 or n_ext == 0:
            y0_mean = np.mean(data.Y_control_int)
            ate = y1_mean - y0_mean
            return EstimationResult(
                ate_est=ate,
                bias=ate - data.true_sate,
                error=(ate - data.true_sate)**2,
                weights_internal=np.ones(n_int),
                weights_external=np.zeros(n_ext)
            )

        # Prepare for Propensity Score Estimation
        # Pool: Internal Controls (1) vs External Controls (0)
        # Note: Some IPW variants pool [Treat+Control_Int] vs External. 
        # Here we assume we are weighting External to match Internal Control (or Target).
        # Based on code: pooling Internal + External.
        X_pool = np.vstack([data.X_control_int, data.X_external])
        source = np.array([1] * n_int + [0] * n_ext)
        
        # Logistic Regression
        prop_model = LogisticRegression(l1_ratio=0, solver='lbfgs')
        prop_model.fit(X_pool, source)
        
        # Predict P(Source=Internal | X)
        p_int = prop_model.predict_proba(X_pool)[:, 1]
        
        # Clip to avoid division by zero
        eps = 1e-6
        p_int = np.clip(p_int, eps, 1 - eps)
        
        # Odds Weights: p / (1-p)
        # This converts distribution from External -> Internal
        weights = p_int / (1 - p_int)
        
        w_int = np.ones(n_int) 
        w_ext = weights[n_int:] # Extract weights for external part

        # Weighted Mean of Control Outcomes
        y0_w_sum = np.sum(data.Y_control_int * w_int) + np.sum(data.Y_external * w_ext)
        w_sum = np.sum(w_int) + np.sum(w_ext)
        
        if w_sum == 0:
            ate = y1_mean 
        else:
            ate = y1_mean - (y0_w_sum / w_sum)
        
        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            error=(ate - data.true_sate)**2,
            weights_internal=w_int,
            weights_external=w_ext
        )