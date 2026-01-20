import numpy as np
import pandas as pd
from generators import DataGenerator
from designs import FixedRatioDesign, EnergyOptimizedDesign
from estimators import IPWEstimator, EnergyMatchingEstimator
from dataclasses import dataclass, field
from typing import List

@dataclass
class SimLog:
    """Stores errors from each simulation run"""
    errors: List[float] = field(default_factory=list)
    biases: List[float] = field(default_factory=list) # Raw difference (Est - True)
    estimates: List[float] = field(default_factory=list)

    def compute_metrics(self):
        errors_arr = np.array(self.errors)
        biases_arr = np.array(self.biases)
        
        mse = np.mean(errors_arr) # This assumes error was squared before storing, or we calc here
        mse = np.mean(biases_arr**2)
        avg_bias = np.mean(biases_arr)
        
        variance = np.var(biases_arr)
        
        return {
            "MSE": mse,
            "Bias^2": avg_bias**2,
            "Variance": variance,
            "Check (Bias^2+Var)": avg_bias**2 + variance
        }

def run_monte_carlo(n_sims=100):

    gen = DataGenerator(dim=5, beta = np.ones(5))
    
    logs = {
        "Fixed_EnergyMatch": SimLog(),
        "Optimized_IPW": SimLog()
    }

    print(f"Starting simulation with {n_sims} iterations...")

    for i in range(n_sims):
        # 1. Generate Data (New samples, same mechanism)
        rct_data = gen.generate_rct_pool(n=500, mean=np.ones(5), var=1.0)
        ext_data = gen.generate_external_pool(n=2000, mean=np.ones(5)*1.2, var=1.5, beta_bias=0.1)

        # --- Method 1: Fixed Ratio + Energy Matching ---
        design_fixed = FixedRatioDesign(treat_ratio=0.5, fixed_n_aug=100)
        split_fixed = design_fixed.split(rct_data, ext_data)
        
        est_match = EnergyMatchingEstimator()
        res_fixed = est_match.estimate(split_fixed)
        
        logs["Fixed_EnergyMatch"].biases.append(res_fixed.bias) # bias here is (est - true_sate)
        logs["Fixed_EnergyMatch"].estimates.append(res_fixed.ate_est)

        # --- Method 2: Energy Optimized Split + IPW ---
        design_opt = EnergyOptimizedDesign()
        split_opt = design_opt.split(rct_data, ext_data)
        
        est_ipw = IPWEstimator()
        res_opt = est_ipw.estimate(split_opt)
        
        logs["Optimized_IPW"].biases.append(res_opt.bias)
        logs["Optimized_IPW"].estimates.append(res_opt.ate_est)

    # Report
    print("\n--- Simulation Results ---")
    results = []
    for method_name, log in logs.items():
        metrics = log.compute_metrics()
        metrics["Method"] = method_name
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    cols = ["Method", "MSE", "Bias^2", "Variance"]
    print(df_results[cols].to_string(index=False))

if __name__ == "__main__":
    run_monte_carlo(n_sims=50)