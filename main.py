import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from structures import EstimationResult
from generators import DataGenerator
from design import FixedRatioDesign, EnergyOptimisedDesign, PooledEnergyMinimizer
from estimator import IPWEstimator, EnergyMatchingEstimator, DummyMatchingEstimator
from dataclasses import dataclass, field
from typing import List, Any


# ----- GLOBAL CONFIG ----- 
N_SIMS = 100
DIM = 5

MEAN_RCT = np.ones(DIM)
VAR_RCT = 1.0

VAR_EXT = 1.5
BIAS_EXT = 1
BETA_BIAS_EXT = 0.

N_RCT = 300
N_EXT = 1000
# -------------------------


# ----- PIPELINES ------
@dataclass
class MethodPipeline:
    name: str
    design: Any 
    estimator: Any

PIPELINES = [
        MethodPipeline(
            name="RCT_ONLY_05",
            design=FixedRatioDesign(treat_ratio=0.5, target_n_aug=0),
            estimator=DummyMatchingEstimator()
        ),
        MethodPipeline(
            name="RCT_ONLY_04",
            design=FixedRatioDesign(treat_ratio=0.4, target_n_aug=0),
            estimator=DummyMatchingEstimator()
        ),
        MethodPipeline(
            name="RCT_ONLY_06",
            design=FixedRatioDesign(treat_ratio=0.6, target_n_aug=0),
            estimator=DummyMatchingEstimator()
        ),
        MethodPipeline(
            name="Dummy_AllExt",
            design=FixedRatioDesign(treat_ratio=0.5, target_n_aug=N_EXT),
            estimator=DummyMatchingEstimator()
        ),
        MethodPipeline(
            name="Dummy_random_10",
            design=FixedRatioDesign(treat_ratio=0.5, target_n_aug=10),
            estimator=DummyMatchingEstimator()
        )
    ]
# ----------------------

@dataclass
class SimLog:
    """Stores full results from each simulation run"""
    results: List[EstimationResult] = field(default_factory=list)

    def compute_metrics(self):
        errors = [r.bias for r in self.results]
        estimates = [r.ate_est for r in self.results]
        n_exts = [r.n_external_used() for r in self.results]
        ess_vals = [r.ess_external() for r in self.results]
        
        errors_arr = np.array(errors)
        
        mse = np.mean(errors_arr**2)
        bias = np.mean(errors_arr)
        variance = np.var(estimates)
        avg_n_ext = np.mean(n_exts)
        avg_ess = np.mean(ess_vals)

        return {
            "MSE": mse,
            "Bias^2": bias**2,
            "Variance": variance,
            "Avg N_Ext": avg_n_ext,
            "Avg ESS_Ext": avg_ess,
            "Check (Bias^2+Var)": bias**2 + variance
        }

def run_single_simulation(seed, dim, beta, n_rct, n_ext, mean_rct, var_rct, var_ext, bias_ext, beta_bias_ext, pipelines):
    """Runs a single iteration of the simulation."""
    # Ensure independent randomness per process
    rng = np.random.default_rng(seed)
    np.random.seed(seed) 

    gen = DataGenerator(dim=dim, beta=beta)
    
    # 1. Generate Data
    rct_data = gen.generate_rct_pool(n=n_rct, mean=mean_rct, var=var_rct)
    ext_data = gen.generate_external_pool(n=n_ext, mean=mean_rct-bias_ext, var=var_ext, beta_bias=beta_bias_ext)

    results = {}
    
    for pipe in pipelines:
        # split
        split_data = pipe.design.split(rct_data, ext_data)
        # estimate
        res = pipe.estimator.estimate(split_data)
        results[pipe.name] = res
    
    return results

def run_monte_carlo(n_sims=100):

    beta_fixed = np.ones(DIM)
        
    logs = {p.name: SimLog() for p in PIPELINES}

    print(f"Starting simulation with {n_sims} iterations (Parallel)...")

    # Generate seeds for reproducibility
    seeds = np.random.randint(0, 1000000, size=n_sims)

    parallel_results = Parallel(n_jobs=10, verbose=5)(
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
            beta_bias_ext=BETA_BIAS_EXT,
            pipelines=PIPELINES
        ) for seed in seeds
    )

    # Aggregate results
    for res_dict in parallel_results:
        for method_name, est_res in res_dict.items():
            logs[method_name].results.append(est_res)

    # Report
    print("\n--- Simulation Results ---")
    results = []
    for method_name, log in logs.items():
        metrics = log.compute_metrics()
        metrics["Method"] = method_name
        results.append(metrics)
    
    df_results = pd.DataFrame(results).sort_values("MSE")
    cols = ["Method", "MSE", "Bias^2", "Variance", "Avg N_Ext", "Avg ESS_Ext"]
    print(df_results[cols].to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    run_monte_carlo(n_sims=N_SIMS)
