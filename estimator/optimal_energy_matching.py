import numpy as np
from structures import SplitData, EstimationResult
from .base import BaseEstimator
from .energy_matching import EnergyMatchingEstimator

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
                 device=None):
        super().__init__()
        self.max_external = max_external
        self.step = step
        self.k_best = k_best
        self.lr = lr
        self.n_iter = n_iter
        self.device = device
        
        # We reuse the matching estimator
        self.matcher = EnergyMatchingEstimator(
            k_best=k_best, 
            lr=lr, 
            n_iter=n_iter, 
            device=device
        )

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:
        """
        Iterates n_ext from 0 to max_external (step size) and finds the best.
        Ignores the `n_external` argument if passed, as this estimator optimizes it.
        """
        
        limit = self.max_external if self.max_external is not None else data.X_external.shape[0]
        
        # If limit is 0, we can't do much, just run with 0
        if limit == 0:
            return self.matcher.estimate(data, n_external=0)

        # Candidates for n_external
        # Ensure we include 0 and potentially the limit
        candidates = list(range(0, limit + 1, self.step))
        if candidates[-1] != limit:
            candidates.append(limit)
            
        best_result = None
        min_energy = float('inf')
        
        for n in candidates:
            # We assume n_external=0 works and returns the baseline energy of using only internal controls
            try:
                res = self.matcher.estimate(data, n_external=n)
                
                # Check energy
                if res.energy_distance < min_energy:
                    min_energy = res.energy_distance
                    best_result = res
            except Exception as e:
                # If n=0 or some other case fails, we log or ignore? 
                # Ideally it shouldn't fail. 
                print(f"Warning: EnergyMatchingEstimator failed for n_external={n}: {e}")
                continue
                
        if best_result is None:
            raise RuntimeError("Could not find any valid estimation result.")
            
        return best_result
