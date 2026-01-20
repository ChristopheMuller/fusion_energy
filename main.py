import numpy as np
from generators import DataGenerator
from designs import FixedRatioDesign, EnergyOptimizedDesign
from estimators import IPWEstimator, EnergyMatchingEstimator

def run_simulation():
    gen = DataGenerator(dim=5)
    rct_data = gen.generate_rct_pool(n=500, mean=np.ones(5), var=1.0)
    ext_data = gen.generate_external_pool(n=2000, mean=np.ones(5)*1.2, var=1.5)

    # Strategy 1: Fixed Ratio Split -> Energy Matching
    print("Running Fixed Ratio Design...")
    design_fixed = FixedRatioDesign(treat_ratio=0.5, fixed_n_aug=100)
    split_data_fixed = design_fixed.split(rct_data, ext_data)
    
    estimator_match = EnergyMatchingEstimator()
    res_fixed = estimator_match.estimate(split_data_fixed)
    print(f"Fixed Design ATE Bias: {res_fixed.bias:.4f}")

    # Strategy 2: Energy Optimized Split -> IPW
    print("Running Energy Optimized Design...")
    design_opt = EnergyOptimizedDesign()
    split_data_opt = design_opt.split(rct_data, ext_data)
    
    estimator_ipw = IPWEstimator()
    res_opt = estimator_ipw.estimate(split_data_opt)
    print(f"Optimized Design ATE Bias: {res_opt.bias:.4f}")
    print(f"Optimized Augmentation N: {split_data_opt.target_n_aug}")

if __name__ == "__main__":
    run_simulation()