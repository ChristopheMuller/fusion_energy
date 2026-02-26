import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Any, Tuple
import torch

from structures import SplitData
from generators import DataGenerator
from design import FixedRatioDesign
from estimator import Optimal_Energy_MatchingEstimator

import os
os.environ["RENV_CONFIG_SANDBOX_ENABLED"] = "FALSE"

# ----- GLOBAL CONFIG ----- 
N_SIMS = 2
DIM = 5
MEAN_RCT = np.ones(DIM)
VAR_RCT = 1.0
CORR = 0.3
VAR_EXT = 1.5
BIAS_EXT = 0.7
BETA_BIAS_EXT = 0.0
NON_LINEAR_COVARIATES = True
NON_LINEAR_OUTCOME = True
N_RCT = 100
N_EXT = 500

B_ITERATIONS = 100
ALPHA = 0.05
POWER_BN = 0.8

DESIGN = FixedRatioDesign(treat_ratio_prior=0.5, target_n_aug=None)
ESTIMATOR = Optimal_Energy_MatchingEstimator(max_external=170, step=1)

# -------------------------

def cate_function(X):
    return 2.0 * np.ones(X.shape[0])
TREATMENT_EFFECT = cate_function

def _subsample_iteration(seed: int, data: SplitData, tau_hat_n: float, 
                         b_n_treat: int, b_n_ctrl: int, b_n_ext: int, 
                         b_n_actual: int, estimator: Any) -> float:
    """Runs a single subsample iteration without replacement and calculates the scaled error."""
    rng = np.random.default_rng(seed)
    
    # 1. Draw without replacement preserving allocation ratio
    idx_treat = rng.choice(len(data.X_treat), b_n_treat, replace=False)
    idx_ctrl = rng.choice(len(data.X_control_int), b_n_ctrl, replace=False)
    idx_ext = rng.choice(len(data.X_external), b_n_ext, replace=False)
    
    sub_data = SplitData(
        X_treat=data.X_treat[idx_treat],
        Y_treat=data.Y_treat[idx_treat],
        X_control_int=data.X_control_int[idx_ctrl],
        Y_control_int=data.Y_control_int[idx_ctrl],
        X_external=data.X_external[idx_ext],
        Y_external=data.Y_external[idx_ext],
        true_sate=data.true_sate,
        target_n_aug=data.target_n_aug
    )
    
    # 2. Run the full pipeline on the subsample
    sub_res = estimator.estimate(sub_data)
    tau_hat_bn = sub_res.ate_est
    
    # 3. Calculate scaled error: sqrt(b_n) * (\hat{\tau}_{b_n}^* - \hat{\tau}_n)
    scaled_error = np.sqrt(b_n_actual) * (tau_hat_bn - tau_hat_n)
    return scaled_error

def compute_subsampling_ci(data: SplitData, estimator: Any, rng: np.random.Generator,
                           B: int = 1000, alpha: float = 0.05, power_bn: float = 0.8, 
                           n_jobs_inner: int = 1) -> Tuple[float, float, float]:
    """
    Constructs Confidence Intervals using the m-out-of-n subsampling procedure.
    """
    # Step 1: Compute full-sample point estimate
    full_res = estimator.estimate(data)
    tau_hat_n = full_res.ate_est
    
    # Calculate pooled n
    n_treat = len(data.X_treat)
    n_ctrl = len(data.X_control_int)
    n_ext = len(data.X_external)
    n_total = n_treat + n_ctrl + n_ext
    
    # Step 2: Determine subsample size b_n (e.g., n^0.8)
    b_n = int(np.floor(n_total ** power_bn))
    
    # Ensure allocation ratio is preserved
    b_n_treat = max(1, int(np.round(b_n * (n_treat / n_total))))
    b_n_ctrl = max(1, int(np.round(b_n * (n_ctrl / n_total))))
    b_n_ext = max(1, int(np.round(b_n * (n_ext / n_total))))
    b_n_actual = b_n_treat + b_n_ctrl + b_n_ext 
    
    print(f"   [Subsampling] Full N = {n_total} | Subsample b_n = {b_n_actual}")

    # Step 3: Run subsampling iterations in parallel
    worker_seeds = rng.integers(0, 2**31 - 1, size=B)
    
    if n_jobs_inner == 1:
        # Run Serially (Outer loop is parallelized)
        scaled_errors = [
            _subsample_iteration(
                seed=worker_seeds[i],
                data=data,
                tau_hat_n=tau_hat_n,
                b_n_treat=b_n_treat,
                b_n_ctrl=b_n_ctrl,
                b_n_ext=b_n_ext,
                b_n_actual=b_n_actual,
                estimator=estimator
            ) for i in range(B)
        ]
    else:
        print(f"   [Subsampling] Running {B} iterations across {n_jobs_inner if n_jobs_inner > 0 else 'all'} cores...")
        scaled_errors = Parallel(n_jobs=n_jobs_inner, verbose=0)(
            delayed(_subsample_iteration)(
                seed=worker_seeds[i],
                data=data,
                tau_hat_n=tau_hat_n,
                b_n_treat=b_n_treat,
                b_n_ctrl=b_n_ctrl,
                b_n_ext=b_n_ext,
                b_n_actual=b_n_actual,
                estimator=estimator
            ) for i in range(B)
        )
    # Step 4: Extract empirical quantiles
    q_lower = np.percentile(scaled_errors, 100 * (alpha / 2))
    q_upper = np.percentile(scaled_errors, 100 * (1 - alpha / 2))
    
    # Step 5: Pivot to find the confidence interval bounds
    ci_lower = tau_hat_n - (q_upper / np.sqrt(n_total))
    ci_upper = tau_hat_n - (q_lower / np.sqrt(n_total))
    
    return tau_hat_n, ci_lower, ci_upper

def run_single_simulation(seed, n_jobs_inner=1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    beta = rng.uniform(1, 3, DIM)
    gen = DataGenerator(dim=DIM, beta=beta, non_linear_covariates=NON_LINEAR_COVARIATES, non_linear_outcome=NON_LINEAR_OUTCOME, rng=rng)
    
    rct_data = gen.generate_rct_pool(n=N_RCT, mean=MEAN_RCT, var=VAR_RCT, corr=CORR, treatment_effect=TREATMENT_EFFECT, rng=rng)
    ext_data = gen.generate_external_pool(n=N_EXT, mean=MEAN_RCT-BIAS_EXT, var=VAR_EXT, corr=CORR, beta_bias=BETA_BIAS_EXT, rng=rng)

    split_data = DESIGN.split(rct_data, ext_data, rng=rng)
    
    print(f"--- Starting Simulation Run (Seed: {seed}) ---")
    tau_hat, ci_lower, ci_upper = compute_subsampling_ci(
        data=split_data, 
        estimator=ESTIMATOR, 
        rng=rng,
        B=B_ITERATIONS, 
        alpha=ALPHA,
        power_bn=POWER_BN,
        n_jobs_inner=n_jobs_inner
    )
    
    # Verify Coverage
    true_ate = split_data.true_sate
    covered = ci_lower <= true_ate <= ci_upper
    
    print(f"   True ATE:  {true_ate:.4f}")
    print(f"   Estimated: {tau_hat:.4f}")
    print(f"   95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Coverage:  {'YES' if covered else 'NO'}\n")
    
    return {
        "True ATE": true_ate,
        "Estimate": tau_hat,
        "CI Lower": ci_lower,
        "CI Upper": ci_upper,
        "Covered": covered
    }

if __name__ == "__main__":
    
    # The Architecture Switch
    if N_SIMS < 10:
        print(f"Low N_SIMS ({N_SIMS}) detected: Parallelizing inner subsampling loop.")
        n_jobs_outer = 1
        n_jobs_inner = -1
    else:
        print(f"High N_SIMS ({N_SIMS}) detected: Parallelizing outer simulation loop.")
        n_jobs_outer = -1
        n_jobs_inner = 1

    seeds = [1234 + i for i in range(N_SIMS)]
    
    # Execution
    if n_jobs_outer == 1:
        # Serial Outer Loop
        results = []
        for seed in seeds:
            res = run_single_simulation(seed=seed, n_jobs_inner=n_jobs_inner)
            results.append(res)
    else:
        # Parallel Outer Loop
        results = Parallel(n_jobs=n_jobs_outer, verbose=10)(
            delayed(run_single_simulation)(seed, n_jobs_inner=n_jobs_inner) for seed in seeds
        )
        
    df = pd.DataFrame(results)
    print(f"\n--- Final Aggregated Results ---")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\nEmpirical Coverage Rate: {df['Covered'].mean() * 100:.1f}%")