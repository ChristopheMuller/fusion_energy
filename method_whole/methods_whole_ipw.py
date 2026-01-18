import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class WholeIPW:
    """
    Estimator (c): IPW on Whole RCT + External data.
    Propensity score is estimated using Logistic Regression with CV.
    Target variable: Treatment indicator (1 for Treated, 0 for Pooled Control).
    """
    def __init__(self, 
                 clip_min=0.01, 
                 clip_max=0.99, 
                 weight_trim_quantile=0.99,
                 cv=5, 
                 max_iter=1000):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.weight_trim_quantile = weight_trim_quantile
        self.weights_ = None
        
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
        
        # 2. Fit Propensity Model
        self.model.fit(X_train, y_train)
        
        # 3. Predict Propensity Scores e(x) = P(Target | X) for the Control pool
        probs = self.model.predict_proba(X_control)[:, 1]
        
        # 4. Clip Probabilities
        probs = np.clip(probs, self.clip_min, self.clip_max)
        
        # 5. Calculate Odds Weights: w = e(x) / (1 - e(x))
        raw_weights = probs / (1 - probs)
        
        # 6. Weight Stabilization
        if self.weight_trim_quantile is not None:
            cutoff = np.quantile(raw_weights, self.weight_trim_quantile)
            raw_weights = np.clip(raw_weights, 0, cutoff)
            
        # 7. Normalization
        self.weights_ = raw_weights / np.sum(raw_weights)
        
        return self

    def estimate(self, X_t, Y_t, X_i, Y_i, X_e, Y_e):
        """
        Fits the IPW model and returns the weighted ATT estimate.
        
        Returns:
            att (float): The estimated Average Treatment Effect on Treated.
            n_observed (int): The number of control units used (RCT Control + External).
        """
        # Fit the model to get weights
        self.fit(X_t, X_i, X_e)
        
        if self.weights_ is None:
            raise RuntimeError("Model fitting failed, weights are None.")

        # Calculate Weighted Mean of Control Outcome
        Y_control_pool = np.concatenate([Y_i, Y_e])
        mu_c_weighted = np.sum(self.weights_ * Y_control_pool)
        
        mu_t = np.mean(Y_t)
        
        att = mu_t - mu_c_weighted
        n_observed = len(Y_control_pool)
        
        return att, n_observed
