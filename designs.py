from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist
from structures import PotentialOutcomes, SplitData

class BaseDesign(ABC):
    @abstractmethod
    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        pass

class FixedRatioDesign(BaseDesign):
    def __init__(self, treat_ratio=0.5, fixed_n_aug=100):
        self.treat_ratio = treat_ratio
        self.fixed_n_aug = fixed_n_aug

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        n_rct = rct_pool.X.shape[0]
        n_treat = int(n_rct * self.treat_ratio)
        
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
            target_n_aug=self.fixed_n_aug
        )

class EnergyOptimizedDesign(BaseDesign):
    def __init__(self, k_folds=5):
        self.k_folds = k_folds

    def _estimate_optimal_augmentation(self, X_rct, X_ext):
        n_ext = X_ext.shape[0]
        # Heuristic: Find k nearest neighbors in energy space or minimal energy subset
        # Placeholder for the actual optimization logic you described
        # calculating the 'effective' sample size that matches the distribution
        
        # For simulation purposes, let's assume we find that 30% of external data is valid
        # In production, this would call the solver
        estimated_n_valid = int(n_ext * 0.3) 
        return estimated_n_valid

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        n_aug = self._estimate_optimal_augmentation(rct_pool.X, ext_pool.X)
        
        n_rct = rct_pool.X.shape[0]        
        n_treat = int((n_rct + n_aug) / 2)
        
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
            target_n_aug=n_aug
        )