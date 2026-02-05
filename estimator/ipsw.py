import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from structures import SplitData, EstimationResult
from .base import BaseEstimator

class IPSWEstimator(BaseEstimator):
    def __init__(self, 
                 clip_min: float = 0.001, 
                 clip_max: float = 0.999, 
                 cv: int = 5,
                 max_iter: int = 1000):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.cv = cv
        self.max_iter = max_iter

    def estimate(self, data: SplitData) -> EstimationResult:
        n_int = data.X_control_int.shape[0]
        n_ext = data.X_external.shape[0]
        n_trt = data.X_treat.shape[0]
        
        y1_mean = np.mean(data.Y_treat)
        
        if n_ext == 0:
            return EstimationResult(
                ate_est=y1_mean - np.mean(data.Y_control_int),
                bias=None, 
                weights_continuous=np.ones(n_int),
                weights_external=np.zeros(n_ext)
            )

        X_pool = np.vstack([data.X_external, data.X_treat])
        source_labels = np.array([0] * n_ext + [1] * n_trt)

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
        
        scores_ext = model.predict_proba(data.X_external)[:, 1]
        scores_ext = np.clip(scores_ext, self.clip_min, self.clip_max)
        
        w_ext = scores_ext / (1 - scores_ext)
        w_int = np.ones(n_int)

        y0_weighted_sum = np.sum(data.Y_control_int * w_int) + np.sum(data.Y_external * w_ext)
        total_weight = np.sum(w_int) + np.sum(w_ext)
        
        y0_weighted_mean = y0_weighted_sum / total_weight
        ate = y1_mean - y0_weighted_mean

        return EstimationResult(
            ate_est=ate,
            bias=ate - data.true_sate,
            weights_continuous=scores_ext,
            weights_external=w_ext
        )