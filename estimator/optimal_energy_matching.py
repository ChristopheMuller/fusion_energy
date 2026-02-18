import numpy as np
import time
import warnings
from structures import SplitData, EstimationResult
from .base import BaseEstimator
from .energy_matching import Energy_MatchingEstimator

class Optimal_Energy_MatchingEstimator(BaseEstimator):
    """
    Finds the optimal number of external controls to add by minimizing Energy Distance.
    It iterates over a range of n_external values and selects the one that yields the lowest Energy Distance.
    """
    def __init__(self, 
                 max_external: int = None, 
                 step: int = 1, 
                 k_best=100, 
                 lr=0.05, 
                 n_iter=1000, 
                 device=None,
                 verbose: bool = False):
        super().__init__()
        self.max_external = max_external
        self.step = step
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.device = device
        self.verbose = verbose
        
        self.matcher = Energy_MatchingEstimator(
            k_best=k_best, 
            lr=lr, 
            n_iter=n_iter, 
            device=device
        )

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        start_time = time.time()
        
        n_available = data.X_external.shape[0]
        
        if self.max_external is not None:
            limit = min(self.max_external, n_available)
        else:
            limit = n_available
            
        if limit == 0:
            if self.verbose: print("Limit is 0, running base estimation.")
            res = self.matcher.estimate(data, n_external=0)
            res.estimation_time = time.time() - start_time
            return res

        safe_step = max(1, int(self.step))

        candidates = list(range(0, limit + 1, safe_step))

        if candidates[-1] < limit:
            candidates.append(limit)
            
        memo = {}

        def get_energy(idx):
            n = candidates[idx]
            if n in memo:
                return memo[n][0]
            
            try:
                res = self.matcher.estimate(data, n_external=n)
                energy = res.energy_distance
                
                if np.isnan(energy):
                    warnings.warn(f"NaN energy detected at n={n}")
                    energy = float('inf')
                    
                memo[n] = (energy, res)
                return energy
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Estimation failed for n_external={n}: {e}")
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
        best_n = -1
        
        for i in range(left, right + 1):
            n = candidates[i]
            
            # Ensure computed (if brute forcing, this computes it now)
            energy = get_energy(i)
                
            if n in memo:
                energy, res = memo[n]
                if energy < min_energy:
                    min_energy = energy
                    best_result = res
                    best_n = n
            
        if best_result is None:
            raise RuntimeError("Optimal Energy Matching failed: All candidate attempts failed or returned infinity.")
            
        best_result.estimation_time = time.time() - start_time
        
        if self.verbose:
            print(f"Optimal found: n_external={best_n} with Energy={min_energy:.6f}")
            
        return best_result