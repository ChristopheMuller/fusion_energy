import numpy as np
from methods.methods_EnergyMatching import EnergyAugmenter_Matching
from utils import compute_energy_distance_numpy

class EnergyRouter:
    """
    Router Method: Selects the optimal n_sampled for EnergyAugmenter_Matching
    by minimizing the Energy Distance between the Augmented Control and the Target.
    
    Implements a smart search strategy (Seed + Refine) to find the minimizer 
    without evaluating every possible integer.
    """
    def __init__(self, n_candidates=None, k_best=1):
        """
        Args:
            n_candidates (list): Optional explicit list of n to try. 
                                 If None, uses smart adaptive search.
            k_best (int): k_best parameter passed to EnergyAugmenter_Matching.
        """
        self.explicit_candidates = n_candidates
        self.k_best = k_best
        self.best_n = None
        self.best_energy = float('inf')
        self._cache = {}

    def _evaluate(self, n, X_t, X_i, X_e, Y_i, Y_e):
        """
        Evaluates a specific n_sampled, caching the result.
        Returns: (energy, (X_w, Y_w))
        """
        if n in self._cache:
            return self._cache[n]
        
        if n == 0:
            en, _, _, _ = compute_energy_distance_numpy(X_i, X_t)
            res = (en, (X_i, Y_i))
            self._cache[n] = res
            return res
            
        # Standard Energy Matching
        augmenter = EnergyAugmenter_Matching(n_sampled=n, k_best=self.k_best, lr=0.01, n_iter=200)
        augmenter.fit(X_t, X_i, X_e)
        
        # Sample to get the matched set
        X_w, Y_w, _ = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
        
        # Compute Energy
        en, _, _, _ = compute_energy_distance_numpy(X_w, X_t)
        
        res = (en, (X_w, Y_w))
        self._cache[n] = res
        return res

    def estimate(self, X_t, Y_t, X_i, Y_i, X_e, Y_e):
        """
        Selects the best n_sampled using a smart search and returns the ATT estimate.
        """
        self._cache = {}
        max_n = len(X_e)
        
        # 1. Determine Seeds
        if self.explicit_candidates:
            seeds = [n for n in self.explicit_candidates if n <= max_n]
        else:
            # Log-linear spacing
            seeds = [0]
            curr = 20
            while curr <= max_n:
                seeds.append(curr)
                curr *= 2
            # Ensure we cover the upper bound if it's far from last seed
            if max_n not in seeds and (max_n - seeds[-1] > 20):
                seeds.append(max_n)
        
        # 2. Evaluate Seeds to find initial best region
        best_n = seeds[0]
        best_en = float('inf')
        
        for n in seeds:
            en, _ = self._evaluate(n, X_t, X_i, X_e, Y_i, Y_e)
            if en is not None and en < best_en:
                best_en = en
                best_n = n
        
        # 3. Refine Search (Adaptive Hill Climbing)
        # Only if we are in smart mode
        if self.explicit_candidates is None:
            # Heuristic: Set initial step size based on magnitude of best_n
            # e.g. if best_n=100, step=50. If best_n=0, step=10.
            step = max(10, best_n // 2)
            current_n = best_n
            
            while step >= 5:
                candidates = [current_n - step, current_n + step]
                improved = False
                
                for cand in candidates:
                    if 0 <= cand <= max_n:
                        en, _ = self._evaluate(cand, X_t, X_i, X_e, Y_i, Y_e)
                        if en is not None and en < best_en:
                            best_en = en
                            best_n = cand
                            current_n = cand
                            improved = True
                
                if not improved:
                    step //= 2
                # If improved, we keep the step size and try to move further in next iter
        
        self.best_n = best_n
        self.best_energy = best_en
        
        # Retrieve optimal result
        _, (X_best, Y_best) = self._evaluate(best_n, X_t, X_i, X_e, Y_i, Y_e)
        
        mu_t = np.mean(Y_t)
        mu_c = np.mean(Y_best)
        
        att = mu_t - mu_c
        n_observed = len(Y_best)
        
        return att, n_observed