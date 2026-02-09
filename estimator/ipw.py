import numpy as np
import torch
import torch.nn.functional as F
import time

from structures import SplitData, EstimationResult
from metrics import optimise_soft_weights, compute_batch_energy
from .base import BaseEstimator

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class IPWEstimator(BaseEstimator):
    
    
    def __init__(self, cv=5, max_iter=1000, clip_max=10.0):
        super().__init__()
        self.cv = cv
        self.max_iter = max_iter
        self.clip_max = clip_max

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()

        x_rct = np.vstack([data.X_treat, data.X_control_int])
        x_ext = data.X_external

        x_combined = np.vstack([x_rct, x_ext])
        s_labels = np.array([1] * len(x_rct) + [0] * len(x_ext))

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
        model.fit(x_combined, s_labels)

        probs_ext = model.predict_proba(x_ext)[:, 1] # probs of ext to be in RCT
        weights_ext = probs_ext / (1 - probs_ext)
        weights_ext = np.clip(weights_ext, 0, self.clip_max)

        y1_mean = np.mean(data.Y_treat)
        sum_y0_weighted = np.sum(data.Y_control_int) + np.sum(data.Y_external * weights_ext)
        total_y0_weight = len(data.Y_control_int) + np.sum(weights_ext)

        y0_fused_mean = sum_y0_weighted / total_y0_weight

        ate_est = y1_mean - y0_fused_mean

        return EstimationResult(
            ate_est = ate_est,
            bias = data.true_sate - ate_est,
            weights_continuous = probs_ext,
            weights_external=weights_ext,
            energy_distance = 0,
            estimation_time=time.time() - start_time
        )