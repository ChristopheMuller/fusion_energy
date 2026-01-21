import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from generators import DataGenerator
from designs import FixedRatioDesign, EnergyOptimizedDesign
from estimators import IPWEstimator, EnergyMatchingEstimator
from dataclasses import dataclass, field
from typing import List


# ----- GLOBAL CONFIG -----
N_SIMS = 100
DIM = 5

MEAN_RCT = np.ones(DIM)
VAR_RCT = 1.0

VAR_EXT = 1.5
BIAS_EXT = 1
BETA_BIAS_EXT = 0.

N_RCT = 500
N_EXT = 1000
# -------------------------

@dataclass
class SimLog:
    """Stores bias from each simulation run"""
    errors: List[float] = field(default_factory=list)
    estimates: List[float] = field(default_factory=list)

    def compute_metrics(self):
        errors_arr = np.array(self.errors)
        
        mse = np.mean(errors_arr**2)
        bias = np.mean(errors_arr)
        variance = np.var(self.estimates)

        return {
            "MSE": mse,
            "Bias^2": bias**2,
            "Variance": variance,
            "Check (Bias^2+Var)": bias**2 + variance
        }

def run_single_simulation(seed, dim, beta, n_rct, n_ext, mean_rct, var_rct, var_ext, bias_ext, beta_bias_ext):
    """Runs a single iteration of the simulation."""
    # Ensure independent randomness per process
    rng = np.random.default_rng(seed)
    # Patch global numpy seed if needed by legacy code in generators (if they use np.random)
    # The current DataGenerator uses np.random directly.
    np.random.seed(seed) 

    gen = DataGenerator(dim=dim, beta=beta)
    
    # 1. Generate Data
    rct_data = gen.generate_rct_pool(n=n_rct, mean=mean_rct, var=var_rct)
    ext_data = gen.generate_external_pool(n=n_ext, mean=mean_rct-bias_ext, var=var_ext, beta_bias=beta_bias_ext)

    results = {}

    # --- Method 0: No Augmentation (Internal Only) ---
    design_no_aug = FixedRatioDesign(treat_ratio=0.5, fixed_n_aug=0)
    split_no_aug = design_no_aug.split(rct_data, ext_data)
    
    est_no_aug = IPWEstimator() 
    res_no_aug = est_no_aug.estimate(split_no_aug)
    results["NoAug_InternalOnly"] = (res_no_aug.bias, res_no_aug.ate_est)

    # --- Method 1: Fixed Augmentation + Energy Matching ---
    design_fixed = FixedRatioDesign(treat_ratio=0.5, fixed_n_aug=100)
    split_fixed = design_fixed.split(rct_data, ext_data)
    
    est_match = EnergyMatchingEstimator()
    res_match = est_match.estimate(split_fixed)
    results["Fixed_EnergyMatch"] = (res_match.bias, res_match.ate_est)

    # --- Method 2: Energy Optimized Split + IPW ---
    design_opt = EnergyOptimizedDesign(n_min=50, n_max=500, k_folds=3, n_iter=200)
    split_opt = design_opt.split(rct_data, ext_data)
    
    est_ipw = IPWEstimator()
    res_ipw = est_ipw.estimate(split_opt)
    results["OptimalN_IPW"] = (res_ipw.bias, res_ipw.ate_est)
    
    return results

def run_monte_carlo(n_sims=100):

    beta_fixed = np.ones(DIM)
    
    logs = {
        "NoAug_InternalOnly": SimLog(),
        "Fixed_EnergyMatch": SimLog(),
        "OptimalN_IPW": SimLog()
    }

    print(f"Starting simulation with {n_sims} iterations (Parallel)...")

    # Generate seeds for reproducibility
    seeds = np.random.randint(0, 1000000, size=n_sims)

    parallel_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_single_simulation)(
            seed=seed,
            dim=DIM,
            beta=beta_fixed,
            n_rct=N_RCT,
            n_ext=N_EXT,
            mean_rct=MEAN_RCT,
            var_rct=VAR_RCT,
            var_ext=VAR_EXT,
            bias_ext=BIAS_EXT,
            beta_bias_ext=BETA_BIAS_EXT
        ) for seed in seeds
    )

    # Aggregate results
    for res in parallel_results:
        for method, (bias, est) in res.items():
            logs[method].errors.append(bias)
            logs[method].estimates.append(est)

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
