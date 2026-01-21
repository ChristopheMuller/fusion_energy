import numpy as np
import pandas as pd
from generators import DataGenerator
from designs import FixedRatioDesign, EnergyOptimizedDesign
from estimators import IPWEstimator, EnergyMatchingEstimator
from dataclasses import dataclass, field
from typing import List


# --- GLOBAL CONFIG ---
N_SIMS = 100
DIM = 5

MEAN_RCT = np.ones(DIM)
VAR_RCT = 1.0

VAR_EXT = 1.5
BIAS_EXT = 1
BETA_BIAS_EXT = 0.

N_RCT = 500
N_EXT = 1000

@dataclass
class SimLog:
    """Stores bias from each simulation run"""
    errors: List[float] = field(default_factory=list)
    estimates: List[float] = field(default_factory=list)

    def compute_metrics(self):
        errors_arr = np.array(self.errors)
        
        mse = np.mean(errors_arr**2)
        avg_bias = np.mean(errors_arr)
        variance = np.var(self.estimates) # Variance of the estimator

        return {
            "MSE": mse,
            "Bias^2": avg_bias**2,
            "Variance": variance,
            "Check (Bias^2+Var)": avg_bias**2 + variance
        }

def run_monte_carlo(n_sims=100):

    gen = DataGenerator(dim=DIM, beta = np.ones(DIM))
    
    logs = {
        "NoAug_InternalOnly": SimLog(),
        "Fixed_EnergyMatch": SimLog(),
        "OptimalN_IPW": SimLog()
    }

    print(f"Starting simulation with {n_sims} iterations...")

    for i in range(n_sims):
        if (i+1) % 10 == 0:
            print(f"  ... iter {i+1}/{n_sims}")
        # 1. Generate Data (New samples, same mechanism)
        rct_data = gen.generate_rct_pool(n=N_RCT, mean=MEAN_RCT, var=VAR_RCT)
        ext_data = gen.generate_external_pool(n=N_EXT, mean=MEAN_RCT-BIAS_EXT, var=VAR_EXT, beta_bias=BETA_BIAS_EXT)

        # --- Method 0: No Augmentation (Internal Only) ---
        design_no_aug = FixedRatioDesign(treat_ratio=0.5, fixed_n_aug=0)
        split_no_aug = design_no_aug.split(rct_data, ext_data)
        
        # IPW estimator with target_n_aug=0 defaults to internal-only mean
        est_no_aug = IPWEstimator() 
        res_no_aug = est_no_aug.estimate(split_no_aug)
        
        logs["NoAug_InternalOnly"].errors.append(res_no_aug.bias)
        logs["NoAug_InternalOnly"].estimates.append(res_no_aug.ate_est)

        # --- Method 1: Fixed Augmentation + Energy Matching ---
        design_fixed = FixedRatioDesign(treat_ratio=0.5, fixed_n_aug=100)
        split_fixed = design_fixed.split(rct_data, ext_data)
        
        est_match = EnergyMatchingEstimator()
        res_match = est_match.estimate(split_fixed)
        
        logs["Fixed_EnergyMatch"].errors.append(res_match.bias)
        logs["Fixed_EnergyMatch"].estimates.append(res_match.ate_est)

        # --- Method 2: Energy Optimized Split + IPW ---
        design_opt = EnergyOptimizedDesign(n_min=50, n_max=500, k_folds=3, n_iter=200)
        split_opt = design_opt.split(rct_data, ext_data)
        
        est_ipw = IPWEstimator()
        res_ipw = est_ipw.estimate(split_opt)
        
        logs["OptimalN_IPW"].errors.append(res_ipw.bias)
        logs["OptimalN_IPW"].estimates.append(res_ipw.ate_est)

    # Report
    print("\n--- Simulation Results ---")
    results = []
    for method_name, log in logs.items():
        metrics = log.compute_metrics()
        metrics["Method"] = method_name
        results.append(metrics)
    
    df_results = pd.DataFrame(results).sort_values("MSE")
    cols = ["Method", "MSE", "Bias^2", "Variance"]
    print(df_results[cols].to_string(index=False))

if __name__ == "__main__":
    run_monte_carlo(n_sims=N_SIMS)
