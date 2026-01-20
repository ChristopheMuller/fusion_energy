import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class IPWAugmenter:
    """
    Robust IPW implementation wrapped in the Augmenter API.

    Improvements over standard implementation:
    - Uses LogisticRegressionCV for automatic regularization (prevents overfitting).
    - Includes StandardScaler (required for regularization to treat features equally).
    - Handles class imbalance via class_weight='balanced'.
    - Implements Weight Trimming (winsorizing) to prevent weight explosion.
    """
    def __init__(self, 
                 clip_min=0.01, 
                 clip_max=0.99, 
                 weight_trim_quantile=0.99,
                 cv=5, 
                 max_iter=1000, 
                 **kwargs):
        """
        Args:
            clip_min (float): Lower bound for propensity scores to avoid div by zero.
            clip_max (float): Upper bound for propensity scores.
            weight_trim_quantile (float): If not None, clips the final weights at this 
                                          quantile (e.g., 0.99) to prevent outliers.
            cv (int): Number of folds for cross-validation in propensity model.
        """
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.weight_trim_quantile = weight_trim_quantile
        self.weights_ = None
        
        # Pipeline: Scaling is crucial for Regularization (L2) to work effectively
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                cv=cv, 
                solver='lbfgs', 
                max_iter=max_iter,
                l1_ratios=(0,),
                class_weight='balanced',
                scoring='neg_log_loss',
                use_legacy_attributes=False
            )
        )

    def fit(self, X_target, X_internal, X_external):
        # 1. Prepare Data
        # Control Pool (Internal + External) -> Label 0
        X_control = np.vstack([X_internal, X_external])
        y_control = np.zeros(X_control.shape[0])
        
        # Target Data -> Label 1
        y_target = np.ones(X_target.shape[0])
        
        X_train = np.vstack([X_control, X_target])
        y_train = np.concatenate([y_control, y_target])
        
        # 2. Fit Propensity Model with CV
        self.model.fit(X_train, y_train)
        
        # 3. Predict Propensity Scores e(x) = P(Target | X) for the Control pool
        # We only need weights for the Control data to make them look like Target
        probs = self.model.predict_proba(X_control)[:, 1]
        
        # 4. Clip Probabilities (Positivity Assumption)
        probs = np.clip(probs, self.clip_min, self.clip_max)
        
        # 5. Calculate Odds Weights: w = e(x) / (1 - e(x))
        # This transforms distribution of Control to resemble Target (ATT weights)
        raw_weights = probs / (1 - probs)
        
        # 6. Weight Stabilization (Trimming/Winsorizing)
        # Extreme weights can destroy variance reduction. We clip the top 1% (or user def).
        if self.weight_trim_quantile is not None:
            cutoff = np.quantile(raw_weights, self.weight_trim_quantile)
            raw_weights = np.clip(raw_weights, 0, cutoff)
            
        # 7. Normalization (Sum to 1)
        # Ensures the weighted sample size is handled correctly in downstream tasks
        self.weights_ = raw_weights / np.sum(raw_weights)
        
        return self

    def sample(self, X_target, X_internal, X_external, Y_internal=None, Y_external=None):
        if self.weights_ is None:
            raise ValueError("Must call fit() before sample()")

        return None, None, self.weights_
    
    def get_internal_weights(self):
        return self.weights_