import pytest
import numpy as np
from structures import SplitData
from design import FixedRatioDesign, EnergyOptimisedDesign

def verify_split_data(split_data: SplitData, n_rct: int, n_ext: int):
    assert isinstance(split_data, SplitData)
    assert len(split_data.X_treat) + len(split_data.X_control_int) == n_rct
    assert len(split_data.Y_treat) == len(split_data.X_treat)
    assert len(split_data.Y_control_int) == len(split_data.X_control_int)
    assert len(split_data.X_external) == n_ext
    assert len(split_data.Y_external) == n_ext
    assert hasattr(split_data, 'true_sate')
    assert hasattr(split_data, 'target_n_aug')

def test_fixed_ratio_design(rct_pool, ext_pool):
    design = FixedRatioDesign(treat_ratio=0.5, target_n_aug=5)
    split_data = design.split(rct_pool, ext_pool)
    n_rct = rct_pool.X.shape[0]
    n_ext = ext_pool.X.shape[0]
    verify_split_data(split_data, n_rct, n_ext)
    assert len(split_data.X_treat) == n_rct // 2
    assert split_data.target_n_aug == 5

def test_energy_optimised_design(rct_pool, ext_pool):
    # Reduced iterations for faster testing
    design = EnergyOptimisedDesign(k_folds=2, n_iter=10, k_best=5, n_max=20)
    split_data = design.split(rct_pool, ext_pool)
    n_rct = rct_pool.X.shape[0]
    n_ext = ext_pool.X.shape[0]
    verify_split_data(split_data, n_rct, n_ext)

