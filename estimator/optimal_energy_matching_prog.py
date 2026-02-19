import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from structures import SplitData, EstimationResult
from .base import BaseEstimator
from .energy_matching import Energy_MatchingEstimator

class OptimalProg_Energy_MatchingEstimator(BaseEstimator):
    def __init__(self, 
                 max_external: int = None, 
                 step: int = 1, 
                 k_best=300, 
                 lr=0.05, 
                 n_iter=1000, 
                 device=None):
        super().__init__()
        self.max_external = max_external
        self.step = step
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.device = device
        
        # We reuse the matching estimator
        self.matcher = Energy_MatchingEstimator(
            k_best=k_best, 
            lr=lr, 
            n_iter=n_iter, 
            device=device
        )

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        """
        Iterates n_ext from 0 to max_external (step size) and finds the best using ternary search.
        Ignores the `n_external` argument if passed, as this estimator optimizes it.
        """
        start_time = time.time()
        
        limit = self.max_external if self.max_external is not None else data.X_external.shape[0]
        
        # If limit is 0, we can't do much, just run with 0
        if limit == 0:
            res = self.matcher.estimate(data, n_external=0)
            res.estimation_time = time.time() - start_time
            return res

        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5,
            oob_score=True,
            n_jobs=1
        )
        rf.fit(data.X_external, data.Y_external)
        
        m_internal_control = rf.predict(data.X_control_int)
        m_external_oob = rf.oob_prediction_
        m_treated = rf.predict(data.X_treat)
        
        estim_YTreated = np.mean(m_treated)

        candidates = list(range(0, limit + 1, self.step))
        if candidates[-1] != limit:
            candidates.append(limit)
            
        memo = {}

        def get_score(idx):
            n = candidates[idx]
            if n in memo:
                return memo[n][0]
            
            try:
                res = self.matcher.estimate(data, n_external=n)
                w_ext = res.weights_external
                
                estim_YControl = np.sum(m_internal_control) + np.sum(w_ext * m_external_oob)
                regul = np.square(estim_YTreated - estim_YControl / (len(m_internal_control) + np.sum(w_ext)))
                
                memo[n] = (regul, res)
                return regul
            except Exception:
                return float('inf')

        left = 0
        right = len(candidates) - 1
        
        while right - left > 2:
            m1 = left + (right - left) // 3
            m2 = right - (right - left) // 3
            e1 = get_score(m1)
            e2 = get_score(m2)

            if e1 < e2:
                right = m2
            else:
                left = m1
                
        best_result = None
        min_score = float('inf')
        
        for i in range(left, right + 1):
            n = candidates[i]
            if n not in memo:
                get_score(i)
                
            if n in memo:
                regul, res = memo[n]
                if regul < min_score:
                    min_score = regul
                    best_result = res
            
        if best_result is None:
            raise RuntimeError("Could not find any valid estimation result.")
            
        best_result.energy_distance = None
        best_result.estimation_time = time.time() - start_time
        return best_result