import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from structures import SplitData, EstimationResult
from .base import BaseEstimator

class IPWEstimator(BaseEstimator):
    """
    Estimates ATT using Inverse Probability Weighting (IPW).
    
    STRATEGY B (Targeting Treated):
    This estimator fits a propensity model to distinguish between Treated (1)
    and External Controls (0). 
    
    Weighting Logic:
    1. Internal Controls: Fixed weight = 1.0 (Assume perfect representation).
    2. External Controls: 
       - Calculate IPW Odds weights to match Treated distribution.
       - Re-normalize these weights so their sum equals 'target_n'.
    """
    def __init__(self, 
                 n_external: int = None,
                 clip_min: float = 0.001, 
                 clip_max: float = 0.999, 
                 cv: int = 5,
                 max_iter: int = 1000):
        """
        Args:
            n_external (int): Optional override for target augmentation size.
            clip_min (float): Min propensity score to avoid instability.
            clip_max (float): Max propensity score.
            cv (int): Cross-validation folds for LogisticRegressionCV.
            max_iter (int): Max iterations for the solver.
        """
        self.n_external = n_external
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.cv = cv
        self.max_iter = max_iter

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        n_trt = data.X_treat.shape[0]
        
        # Determine target_n (Legacy support for manual override)
        if n_external is not None:
            target_n = n_external
        elif self.n_external is not None:
            target_n = int(self.n_external)
        else:
            target_n = data.target_n_aug

        y1_mean = np.mean(data.Y_treat)
        
        # Edge case: No augmentation requested
        if target_n == 0 or n_ext == 0:
            return EstimationResult(
                ate_est=y1_mean - np.mean(data.Y_control_int),
                bias=None, # Cannot compute without true SATE specific to this split
                weights_internal=np.ones(n_int),
                weights_external=np.zeros(n_ext)
            )

        # 1. Prepare Data for Propensity Model
        # Train on External (0) vs Treated (1)
        X_pool = np.vstack([data.X_external, data.X_treat])
        source_labels = np.array([0] * n_ext + [1] * n_trt)
        
        # 2. Fit Robust Propensity Model
        # Pipeline: Scale features -> LogRegCV (Auto-tune L2 penalty)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                cv=self.cv,
                solver='lbfgs',
                max_iter=self.max_iter,
                class_weight='balanced',
                scoring='neg_log_loss',
                l1_ratios=(0,),
                use_legacy_attributes=False
            )
        )
        model.fit(X_pool, source_labels)
        
        # 3. Predict Propensity Scores P(Treated | X) for EXTERNAL units
        scores_ext = model.predict_proba(data.X_external)[:, 1]
        
        # 4. Clip Probabilities (Positivity Assumption)
        scores_ext = np.clip(scores_ext, self.clip_min, self.clip_max)
        
        # 5. Calculate Raw IPW Weights (Odds)
        # Formula: p / (1-p) transforms distribution from External -> Treated
        raw_odds = scores_ext / (1 - scores_ext)
        
        # 6. Normalize Weights to sum to target_n
        sum_odds = np.sum(raw_odds)
        
        if sum_odds > 0:
            # Rescale: (Weight / Sum) * Target_Total
            w_ext = (raw_odds / sum_odds) * target_n
        else:
            # Fallback (should not happen with clipping)
            w_ext = np.zeros(n_ext)
                    
        # 7. Internal Weights (Fixed at 1.0)
        w_int = np.ones(n_int)

        # 8. Compute Weighted Mean of Control Outcomes
        # Weighted Average = (Sum(Y_int*1) + Sum(Y_ext*w_ext)) / (N_int + Sum(w_ext))
        # Note: Sum(w_ext) is exactly target_n now.
        y0_weighted_sum = np.sum(data.Y_control_int * w_int) + np.sum(data.Y_external * w_ext)
        total_weight = n_int + target_n # Exact sum
        
        y0_weighted_mean = y0_weighted_sum / total_weight
        ate = y1_mean - y0_weighted_mean

        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_internal=w_int,
            weights_external=w_ext
        )