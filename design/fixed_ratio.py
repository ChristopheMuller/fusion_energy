import numpy as np
from structures import PotentialOutcomes, SplitData
from .base import BaseDesign

class FixedRatioDesign(BaseDesign):
    """
    A design that splits the RCT data according to a fixed ratio (e.g., 1:1)
    and targets a pre-determined fixed number of external samples.
    """
    def __init__(self, treat_ratio=0.5, target_n_aug=100):
        self.treat_ratio = treat_ratio
        self.target_n_aug = target_n_aug

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
            target_n_aug=self.target_n_aug
        )