import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from structures import PotentialOutcomes, SplitData
from .base import BaseDesign

class IPWBalanceDesign(BaseDesign):
    """
    Determines the augmentation size (N_aug) by calculating the sum of 
    propensity score weights (Effective Sample Size) of the external data.

    Methodology:
    1. Fit a Propensity Model: P(Is_RCT | X).
    2. Calculate Odds Weights for External units: w = p / (1-p).
    3. Sum(Weights) -> Effective number of RCT-compatible external units.
    4. Set N_aug = Sum(Weights).
    5. Split RCT to balance final arms: N_treat = (N_rct + N_aug) / 2.
    """
    def __init__(self, 
                 clip_min=0.01, 
                 clip_max=0.99, 
                 cv=5, 
                 max_iter=1000,
                 ratio_trt_after_augmentation=0.5,
                 ratio_trt_before_augmentation=None):
        
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.cv = cv
        self.max_iter = max_iter
        self.ratio_trt_after_augmentation = ratio_trt_after_augmentation
        self.ratio_trt_before_augmentation = ratio_trt_before_augmentation

        if self.ratio_trt_before_augmentation is not None and self.ratio_trt_after_augmentation is not None:
            raise ValueError("Specify only one of ratio_trt_before_augmentation or ratio_trt_after_augmentation.")

    def _estimate_effective_sample_size(self, X_rct, X_ext):
        n_rct = X_rct.shape[0]
        n_ext = X_ext.shape[0]
        
        # 1. Prepare Data: RCT (1) vs External (0)
        X_pool = np.vstack([X_rct, X_ext])
        y_pool = np.array([1] * n_rct + [0] * n_ext)
        
        # 2. Fit Robust Propensity Model
        model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                cv=self.cv,
                solver='lbfgs', 
                max_iter=self.max_iter,
                class_weight='balanced',
                scoring='neg_log_loss',
                use_legacy_attributes=False,
                l1_ratios=(0,)
            )
        )
        model.fit(X_pool, y_pool)
        
        # 3. Predict P(RCT | X) for External units
        scores_ext = model.predict_proba(X_ext)[:, 1]
        
        # 4. Clip and Weight
        scores_ext = np.clip(scores_ext, self.clip_min, self.clip_max)
        
        # 5. Calculate "Pseudo-Count"
        # We cap the effective size at the actual N_ext to be safe
        n_effective = np.sum(scores_ext)
        
        return int(min(n_effective, n_ext))

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        # 1. Estimate Effective N based on compatibility (IPW)
        best_n_aug = self._estimate_effective_sample_size(rct_pool.X, ext_pool.X)
        
        # 2. Perform Design Split
        n_rct = rct_pool.X.shape[0]
        
        # Calculation: N_treat = (N_rct + N_aug) / 2
        if self.ratio_trt_before_augmentation is not None:
            n_treat = int((n_rct) * self.ratio_trt_before_augmentation)
        else:
            n_treat = int((n_rct + best_n_aug) * self.ratio_trt_after_augmentation)
        
        # Safety bounds
        n_treat = min(max(n_treat, 10), n_rct - 10)

        indices = np.random.permutation(n_rct)
        idx_t = indices[:n_treat]
        idx_c = indices[n_treat:]
        
        true_sate = np.mean(rct_pool.Y1[indices] - rct_pool.Y0[indices])

        return SplitData(
            X_treat=rct_pool.X[idx_t],
            Y_treat=rct_pool.Y1[idx_t],
            X_control_int=rct_pool.X[idx_c],
            Y_control_int=rct_pool.Y0[idx_c],
            X_external=ext_pool.X,
            Y_external=ext_pool.Y0,
            true_sate=true_sate,
            target_n_aug=best_n_aug
        )