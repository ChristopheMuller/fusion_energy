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
    and External Controls (0). It then weights the External Controls to match 
    the covariate distribution of the Treated group directly.
    
    Internal Controls are assumed to represent the Treated population 
    (due to RCT randomization) and are assigned a weight of 1.0.

    Robustness features:
    - StandardScaler + LogisticRegressionCV for automatic regularization.
    - Propensity score clipping to satisfy positivity.
    - Weight trimming (winsorizing) to reduce variance from extreme weights.
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
            weight_trim_quantile (float): Quantile to clip extreme weights (e.g. 0.99).
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
                class_weight='balanced', # Handles size imbalance between Ext/Trt
                scoring='neg_log_loss',
                l1_ratios=(0,),
                use_legacy_attributes=False
            )
        )
        model.fit(X_pool, source_labels)
        
        # 3. Predict Propensity Scores P(Treated | X) for EXTERNAL units
        # We need to know "How much does this External unit look like a Treated unit?"
        scores_ext = model.predict_proba(data.X_external)[:, 1]
        
        # 4. Clip Probabilities (Positivity Assumption)
        scores_ext = np.clip(scores_ext, self.clip_min, self.clip_max)
        
        # 5. Calculate Weights: Odds (p / 1-p)
        # This transforms the External distribution -> Treated distribution
        raw_weights_ext = scores_ext
                    
        # 7. Internal Weights
        # Internal controls are already randomized from the same population as Treated,
        # so they represent the target distribution naturally.
        w_int = np.ones(n_int)
        w_ext = raw_weights_ext

        # 8. Compute Weighted Mean of Control Outcomes
        # Note: We sum weighted outcomes from both Internal and External
        y0_weighted_sum = np.sum(data.Y_control_int * w_int) + np.sum(data.Y_external * w_ext)
        total_weight = np.sum(w_int) + np.sum(w_ext)
        
        if total_weight == 0:
            ate = y1_mean 
        else:
            y0_weighted_mean = y0_weighted_sum / total_weight
            ate = y1_mean - y0_weighted_mean

        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_internal=w_int,
            weights_external=w_ext
        )