import numpy as np
import time
from structures import SplitData, EstimationResult
from .base import BaseEstimator
from .energy_matching_prog import EnergyMatchingProgEstimator

class OptimalEnergyMatchingEstimator(BaseEstimator):
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
        self.matcher = EnergyMatchingProgEstimator(
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
                res = self.matcher.estimate(data, n_external=n)
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
