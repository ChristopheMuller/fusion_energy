import numpy as np
import torch
import time
from sklearn.ensemble import RandomForestRegressor
from structures import SplitData, EstimationResult
from .base import BaseEstimator
from .MSEProg_matching import MSEProg_MatchingEstimator

class Optimal_MSEProg_MatchingEstimator(BaseEstimator):
    """
    Finds the optimal number of external controls to add by minimizing Energy Distance.
    It iterates over a range of n_external values and selects the one that yields the lowest Energy Distance.
    """
    def __init__(self, 
                 max_external: int = None, 
                 step: int = 1, 
                 k_best=100, 
                 lr=0.05, 
                 n_iter=300, 
                 lamb=1.0,
                 device=None):
        super().__init__()
        self.max_external = max_external
        self.step = step
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.device = device
        self.lamb = lamb
        
        # We reuse the matching estimator
        self.matcher = MSEProg_MatchingEstimator(
            k_best=k_best, 
            lr=lr, 
            n_iter=n_iter, 
            lamb=lamb,
            device=device
        )

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        """
        Iterates n_ext from 0 to max_external (step size) and finds the best using ternary search.
        Ignores the `n_external` argument if passed, as this estimator optimizes it.
        """
        start_time = time.time()

        # 1. Precompute Tensors and Distance Matrices
        device = self.device if self.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_t = torch.as_tensor(data.X_treat, dtype=torch.float32, device=device)
        X_c = torch.as_tensor(data.X_control_int, dtype=torch.float32, device=device)
        X_e = torch.as_tensor(data.X_external, dtype=torch.float32, device=device)

        dist_st = torch.cdist(X_e, X_t)
        dist_ss = torch.cdist(X_e, X_e)
        dist_is = torch.cdist(X_c, X_e)

        dist_st_sum = dist_st.sum(dim=1)
        dist_is_sum = dist_is.sum(dim=0)

        sum_it = torch.cdist(X_c, X_t).sum()
        sum_ii = torch.cdist(X_c, X_c).sum()

        # Precompute prognostic scores
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            oob_score=True,
            n_jobs=1
        )
        rf.fit(data.X_external, data.Y_external)
        m_c = torch.as_tensor(rf.predict(data.X_control_int), dtype=torch.float32, device=device)
        m_e = torch.as_tensor(rf.oob_prediction_, dtype=torch.float32, device=device)
        m_t = torch.as_tensor(rf.predict(data.X_treat), dtype=torch.float32, device=device)

        precomputed = {
            'X_t': X_t,
            'X_c': X_c,
            'X_e': X_e,
            'dist_st': dist_st,
            'dist_ss': dist_ss,
            'dist_is': dist_is,
            'dist_st_sum': dist_st_sum,
            'dist_is_sum': dist_is_sum,
            'sum_it': sum_it,
            'sum_ii': sum_ii,
            'm_c': m_c,
            'm_e': m_e,
            'm_t': m_t
        }
        
        limit = self.max_external if self.max_external is not None else data.X_external.shape[0]
        
        # If limit is 0, we can't do much, just run with 0
        if limit == 0:
            res = self.matcher.estimate(data, n_external=0)
            res.estimation_time = time.time() - start_time
            return res

        # Candidates for n_external
        # Ensure we include 0 and potentially the limit
        candidates = list(range(0, limit + 1, self.step))
        if candidates[-1] != limit:
            candidates.append(limit)
            
        memo = {}

        def get_energy(idx):
            n = candidates[idx]
            if n in memo:
                return memo[n][0]
            
            try:
                res = self.matcher.estimate(data, n_external=n, **precomputed)
                energy = res.energy_distance
                memo[n] = (energy, res)
                return energy
            except Exception as e:
                print(f"Warning: EnergyMatchingEstimator failed for n_external={n}: {e}")
                return float('inf')

        # Ternary Search for U-shaped energy curve
        left = 0
        right = len(candidates) - 1
        
        while right - left > 2:
            m1 = left + (right - left) // 3
            m2 = right - (right - left) // 3
            e1 = get_energy(m1)
            e2 = get_energy(m2)
            
            if e1 < e2:
                right = m2
            else:
                left = m1
                
        best_result = None
        min_energy = float('inf')
        
        for i in range(left, right + 1):
            n = candidates[i]
            # Ensure computed
            if n not in memo:
                get_energy(i)
                
            if n in memo:
                energy, res = memo[n]
                if energy < min_energy:
                    min_energy = energy
                    best_result = res
            
        if best_result is None:
            raise RuntimeError("Could not find any valid estimation result.")
            
        best_result.estimation_time = time.time() - start_time
        return best_result
