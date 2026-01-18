import numpy as np
from methods.methods_EnergyMatching import EnergyAugmenter_Matching
from utils import compute_energy_distance_numpy

class EnergyRouter:
    """
    Router Method: Selects the optimal n_sampled for EnergyAugmenter_Matching
    by minimizing the Energy Distance between the Augmented Control and the Target.
    """
    def __init__(self, n_candidates=None, k_best=1):
        """
        Args:
            n_candidates (list): List of n_sampled values to try. 
                                 If None, defaults to [0, 10, 20, 30, 50, 75, 100, 150].
            k_best (int): k_best parameter passed to EnergyAugmenter_Matching.
        """
        if n_candidates is None:
            self.n_candidates = [0, 10, 20, 30, 50, 75, 100, 150]
        else:
            self.n_candidates = n_candidates
        self.k_best = k_best
        self.best_n = None
        self.best_augmenter = None
        self.best_energy = float('inf')

    def estimate(self, X_t, Y_t, X_i, Y_i, X_e, Y_e):
        """
        Selects the best n_sampled and returns the ATT estimate.
        
        Returns:
            att (float): The estimated Average Treatment Effect on Treated.
            n_observed (int): The number of control units used (RCT Control + n_sampled).
        """
        best_energy = float('inf')
        best_n = self.n_candidates[0]
        best_result = None # Stores (X_w, Y_w, weights)
        
        # Iterate through candidates
        for n in self.n_candidates:
            if n > len(X_e):
                continue
                
            # Instantiate and fit
            # Note: n_sampled=0 is a special case. 
            # If n_sampled=0, EnergyAugmenter_Matching might behave differently or we handle it here.
            # Looking at EnergyAugmenter_Matching implementation:
            # If n_sampled=0, beta=0. Loop runs but beta=0 means only term2_ie might matter?
            # Actually, if n=0, we just use RCT Control.
            
            if n == 0:
                # Naive RCT case
                en, _, _, _ = compute_energy_distance_numpy(X_i, X_t)
                current_energy = en
                
                # Store result as if it came from sample()
                # X_w = X_i, Y_w = Y_i. weights logic handled later or simplified here.
                # But to keep consistent interface, let's just compute Energy.
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_n = 0
                    best_result = (X_i, Y_i)
                continue

            augmenter = EnergyAugmenter_Matching(n_sampled=n, k_best=self.k_best, lr=0.01, n_iter=200)
            augmenter.fit(X_t, X_i, X_e)
            
            # Sample to get the matched set
            X_w, Y_w, _ = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
            
            # Compute Energy Distance between Augmented Control (X_w) and Target (X_t)
            # X_w is (RCT Control + Selected External)
            en, _, _, _ = compute_energy_distance_numpy(X_w, X_t)
            
            if en is not None and en < best_energy:
                best_energy = en
                best_n = n
                best_result = (X_w, Y_w)
        
        self.best_n = best_n
        self.best_energy = best_energy
        
        # Calculate ATT using the best result
        X_best, Y_best = best_result
        
        mu_t = np.mean(Y_t)
        mu_c = np.mean(Y_best)
        
        att = mu_t - mu_c
        n_observed = len(Y_best) # This is n_i + best_n
        
        return att, n_observed
