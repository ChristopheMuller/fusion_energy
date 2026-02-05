import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from structures import EstimationResult
from generators import DataGenerator
from design import FixedRatioDesign, EnergyOptimisedDesign, PooledEnergyMinimizer, IPSWBalanceDesign
from estimator import IPWEstimator, EnergyMatchingEstimator, DummyMatchingEstimator, EnergyWeightingEstimator, OptimalEnergyMatchingEstimator
from dataclasses import dataclass, field
from typing import List, Any

from metrics import compute_weighted_energy
from visualisations import (
    plot_error_boxplots, 
    plot_pca_weights, 
    plot_mse_decomposition, 
    plot_energy_distance,
    plot_metric_curves,
    plot_weight_ranks
)

# ----- GLOBAL CONFIG ----- 
N_SIMS = 100
DIM = 2

MEAN_RCT = np.ones(DIM)
VAR_RCT = 1.0
CORR = 0.3

VAR_EXT = 1.5
BIAS_EXT = 0.7      # Mean shift in external data
BETA_BIAS_EXT = 0.0 # Coefficient shift in external data

N_RCT = 200
N_EXT = 1000

# -------------------------


# ----- CATE FUN -----
def cate_function(X):
    """Defines the Treatment Effect as a function of covariates X."""
    return 2.0 * np.ones(X.shape[0])
TREATMENT_EFFECT = cate_function
# --------------------


# ----- PIPELINES ------
@dataclass
class MethodPipeline:
    name: str
    design: Any 
    estimator: Any

PIPELINES = [
        MethodPipeline(
            name="EnergyMatching_NoExt",
            design=FixedRatioDesign(treat_ratio_prior=0.5, target_n_aug=0),
            estimator=DummyMatchingEstimator()
        ),
        MethodPipeline(
            name="IPW",
            design=FixedRatioDesign(treat_ratio_prior=0.5, target_n_aug=0),
            estimator=IPWEstimator()
        ),
        MethodPipeline(
            name="EnergyMatching_OPT_Estimator",
            design=FixedRatioDesign(treat_ratio_prior=0.5, target_n_aug=1),
            estimator=OptimalEnergyMatchingEstimator(step=3, k_best=50, max_external=150)
        ),
        MethodPipeline(
            name="EnergyMatching_OPT_Design_Estimator",
            design=EnergyOptimisedDesign(),
            estimator=OptimalEnergyMatchingEstimator(step=3, k_best=50, max_external=150)
        ),
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
        sum_w_ext = [r.sum_of_weights_external() for r in self.results]
        
        errors_arr = np.array(errors)
        
        mse = np.mean(errors_arr**2)
        bias = np.mean(errors_arr)
        variance = np.var(estimates)
        avg_n_ext = np.mean(n_exts)
        avg_sum_w_ext = np.mean(sum_w_ext)
        avg_energy = np.mean([r.energy_distance for r in self.results])

        return {
            "MSE": mse,
            "Bias^2": bias**2,
            "Variance": variance,
            "Avg N_Ext": avg_n_ext,
            "Avg Sum W_Ext": avg_sum_w_ext,
            "Avg Energy": avg_energy,
            "Check (Bias^2+Var)": bias**2 + variance
        }

def run_single_simulation(seed, dim, n_rct, n_ext, mean_rct, var_rct, var_ext, bias_ext, beta_bias_ext, corr, treatment_effect, pipelines):
    """Runs a single iteration of the simulation."""
    # Ensure independent randomness per process
    rng = np.random.default_rng(seed)
    np.random.seed(seed) 

    beta = rng.uniform(1, 3, dim)

    gen = DataGenerator(dim=dim, beta=beta)
    
    # 1. Generate Data
    rct_data = gen.generate_rct_pool(n=n_rct, mean=mean_rct, var=var_rct, corr=corr, treatment_effect=treatment_effect)
    ext_data = gen.generate_external_pool(n=n_ext, mean=mean_rct-bias_ext, var=var_ext, corr=corr, beta_bias=beta_bias_ext)

    results = {}
    
    # Generate a specific seed for splitting to ensure consistency across methods
    # We use a large integer derived from the main rng
    split_seed = rng.integers(0, 2**63 - 1)
    
    for pipe in pipelines:
        # Create a fresh RNG for this split call to ensure identical splitting behavior
        # across all pipelines (provided the design parameters like ratio are compatible).
        split_rng = np.random.default_rng(split_seed)
        
        # split
        split_data = pipe.design.split(rct_data, ext_data, rng=split_rng)
        # estimate
        res = pipe.estimator.estimate(split_data)
        
        # Metric: Energy Distance (Target vs Pooled Control)
        res.energy_distance = compute_weighted_energy(
            X_target=split_data.X_treat,
            X_internal=split_data.X_control_int,
            X_external=split_data.X_external,
            weights_external=res.weights_external
        )
        
        results[pipe.name] = res
    
    return results

def run_monte_carlo(n_sims=100, seed=None):
        
    logs = {p.name: SimLog() for p in PIPELINES}

    print(f"Starting simulation with {n_sims} iterations (Parallel)...")

    # Generate seeds for reproducibility
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000000, size=n_sims)

    parallel_results = Parallel(n_jobs=10, verbose=5)(
        delayed(run_single_simulation)(
            seed=seed,
            dim=DIM,
            n_rct=N_RCT,
            n_ext=N_EXT,
            mean_rct=MEAN_RCT,
            var_rct=VAR_RCT,
            var_ext=VAR_EXT,
            bias_ext=BIAS_EXT,
            beta_bias_ext=BETA_BIAS_EXT,
            corr=CORR,
            treatment_effect=TREATMENT_EFFECT,
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
    cols = ["Method", "MSE", "Bias^2", "Variance", "Avg N_Ext", "Avg Sum W_Ext", "Avg Energy"]
    print(df_results[cols].to_string(index=False, float_format="%.3f"))

    # Visualisations of results
    plot_error_boxplots(logs)
    plot_mse_decomposition(logs)
    plot_energy_distance(logs)
    plot_metric_curves(logs)

    # Visualisations of example data
    print("Generating PCA plots...")
    
    # Generate one example dataset (using a fixed seed for consistency in plots)
    np.random.seed(42)
    beta_fixed = np.ones(DIM)
    gen = DataGenerator(dim=DIM, beta=beta_fixed)
    
    rct_data_plot = gen.generate_rct_pool(n=N_RCT, mean=MEAN_RCT, var=VAR_RCT, corr=CORR, treatment_effect=TREATMENT_EFFECT)
    ext_data_plot = gen.generate_external_pool(n=N_EXT, mean=MEAN_RCT-BIAS_EXT, var=VAR_EXT, corr=CORR, beta_bias=BETA_BIAS_EXT)
    
    for pipe in PIPELINES:
        rng_plot = np.random.default_rng(42)
        split_data = pipe.design.split(rct_data_plot, ext_data_plot, rng=rng_plot)
        est_result = pipe.estimator.estimate(split_data)
        
        plot_filename = f"plots/{pipe.name}/pca_weights.png"
        plot_pca_weights(split_data, est_result, f"PCA Weights - {pipe.name}", plot_filename)
        
        rank_filename = f"plots/{pipe.name}/weights_rank.png"
        n_ext = split_data.target_n_aug
        plot_weight_ranks(est_result, f"Weight Ranks - {pipe.name}", rank_filename, n_external=n_ext)


if __name__ == "__main__":
    run_monte_carlo(n_sims=N_SIMS, seed=1234)
