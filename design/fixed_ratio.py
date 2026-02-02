from typing import Optional
import numpy as np
from structures import PotentialOutcomes, SplitData
from .base import BaseDesign

class FixedRatioDesign(BaseDesign):
    """
    A design that splits the RCT data according to a fixed ratio (e.g., 1:1)
    and targets a pre-determined fixed number of external samples.
    """
    def __init__(self, treat_ratio_prior=None, treat_ratio_post=None, target_n_aug=100, seed_split=None):
        self.treat_ratio_prior = treat_ratio_prior
        self.treat_ratio_post = treat_ratio_post
        self.target_n_aug = target_n_aug

        # only one of treat_ratio and treat_ratio_after should be set
        if (treat_ratio_prior is None) and (treat_ratio_post is None):
            treat_ratio_prior = 0.5
        assert (treat_ratio_prior is not None) != (treat_ratio_post is not None), "Only one of treat_ratio and treat_ratio_after should be set."

        if seed_split is not None:
            self.rng = np.random.default_rng(seed_split)
        else:
            self.rng = np.random.default_rng()

    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes, rng: Optional[np.random.Generator] = None) -> SplitData:
        n_rct = rct_pool.X.shape[0]

        if self.treat_ratio_post is not None:
            n_treat = int((n_rct + self.target_n_aug) * self.treat_ratio_post)
            n_treat = min(n_treat, n_rct - 5)
        elif self.treat_ratio_prior is not None:
            n_treat = int(n_rct * self.treat_ratio_prior)

        # Use provided rng if available, else use internal
        local_rng = rng if rng is not None else self.rng
        
        indices = local_rng.permutation(n_rct)
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