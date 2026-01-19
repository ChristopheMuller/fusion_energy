import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from data_gen import generate_covariates, generate_outcomes_nonlinear
from methods import EnergyAugmenter_PooledTarget, EnergyAugmenter_Weighting
from utils import compute_energy_distance_numpy
from visualization import (
    plot_bias_variance_comparison,
    plot_energy_comparison,
    plot_covariate_densities,
    plot_outcome_densities,
    plot_covariate_2d_scatter,
    plot_error_boxplot,
    plot_energy_mse_method_decomposition
)

# --- Global Configuration ---
# Only keeping the requested method for this experiment
METHODS_CONFIG = {
    "En.PooledTarget_100k": (EnergyAugmenter_PooledTarget, {'k_best': 100}),
    "En.Weighting": (EnergyAugmenter_Weighting, {})
}

def generate_tau():
    return 2.5

def process_repetition(rep_id, n_sampled_list, n_rct, n_ext, dim, rct_bias, ext_bias, rct_var, ext_var):
    """
    Runs a single simulation repetition under the new Experimental Setup.
    1. Generate RCT population (X_rct) and External population (X_ext).
    2. Loop through n_sampled_list.
       a. Find matching set from X_ext to X_rct (size n_samp).
       b. Determine split sizes: n_treated = (n_rct + n_samp)/2.
       c. Randomly split X_rct into Treatment and Control.
       d. Combine RCT Control + Matched External.
       e. Compute ATE and stats.
    """
    
    # 1. Generate Data
    tau = generate_tau()
    
    # RCT Population (Single population, centered at 1.0)
    mean_rct = np.ones(dim)
    X_rct = generate_covariates(n_rct, dim, mean_rct, var=rct_var)
    
    # External Population (Centered at 1.0 - ext_bias)
    mean_ext = mean_rct - ext_bias
    X_ext = generate_covariates(n_ext, dim, mean_ext, var=ext_var)
    
    # Outcomes
    # We need potential outcomes for RCT to allow post-hoc splitting
    beta = np.ones(dim)
    Y0_rct, Y1_rct = generate_outcomes_nonlinear(X_rct, tau, beta=beta)
    
    # For External, assume Control outcomes (Y0)
    Y0_ext, _ = generate_outcomes_nonlinear(X_ext, tau, beta=beta)
    
    rep_results = []
    
    for n_samp in n_sampled_list:
        
        # Calculate split sizes
        # n_treated = (n_rct + n_samp) / 2
        # We ensure integer sizes
        n_treated = int((n_rct + n_samp) / 2)
        n_control_rct = n_rct - n_treated
        
        # Check validity
        if n_control_rct < 0:
            # Cannot satisfy required treatment size from RCT
            continue

        # Shuffle RCT for random assignment
        perm = np.random.permutation(n_rct)
        idx_t = perm[:n_treated]
        idx_c = perm[n_treated:]
        
        X_t = X_rct[idx_t]
        Y_t = Y1_rct[idx_t]
        
        X_c_rct = X_rct[idx_c]
        Y_c_rct = Y0_rct[idx_c]
        
        for method_name, (MethodClass, kwargs) in METHODS_CONFIG.items():
            
            X_match = None
            Y_match = None
            weights_match = None
            
            if n_samp > 0:
                # Fit and Sample
                # Target: Whole RCT
                # Control: Empty (we are matching to the target directly)
                X_dummy_ctrl = np.empty((0, dim))
                Y_dummy_ctrl = np.empty((0,))
                
                augmenter = MethodClass(n_sampled=n_samp, lr=0.01, n_iter=200, **kwargs)
                augmenter.fit(X_rct, X_dummy_ctrl, X_ext)
                
                # Sample returns fused [internal; external]
                # internal is empty, so it returns external subset
                # X_match should be the "augmented" part
                
                if method_name == "En.Weighting":
                    # Weighting returns full X_ext with weights
                    X_match, Y_match, weights_match = augmenter.sample(
                        X_rct, X_dummy_ctrl, X_ext, Y_dummy_ctrl, Y0_ext
                    )
                else:
                     # Matching methods return a subset (X_fused is just external if internal is empty)
                    X_match, Y_match, weights_match = augmenter.sample(
                        X_rct, X_dummy_ctrl, X_ext, Y_dummy_ctrl, Y0_ext
                    )
            else:
                X_match = np.empty((0, dim))
                Y_match = np.empty((0,))
                weights_match = np.empty((0,))
            
            # Combine Control Group
            X_control_final = np.vstack([X_c_rct, X_match])
            Y_control_final = np.concatenate([Y_c_rct, Y_match])
            
            # --- WEIGHTS CONSTRUCTION ---
            # RCT Control: Assumed weight 1 (unweighted baseline)
            # External (X_match): Use returned weights or default to 1
            
            n_c = X_c_rct.shape[0]
            n_m = X_match.shape[0]
            
            # Base weights for RCT Control
            w_c = np.ones(n_c)
            
            if n_m > 0:
                if method_name == "En.Weighting":
                    # For weighting, weights_match sum to 1.
                    # We want them to represent an "effective sample size" of n_samp
                    # So we scale them by n_samp
                    w_m = weights_match * n_samp
                else:
                    # For matching (PooledTarget), weights_match (from sample) might be messy
                    # But since we selected n_samp units explicitly, we can treat them as weight 1
                    w_m = np.ones(n_m)
            else:
                w_m = np.empty((0,))
            
            weights_final = np.concatenate([w_c, w_m])
            
            # Normalize weights to avoid scale issues in weighted mean?
            # Weighted mean = sum(w*y) / sum(w)
            # So absolute scale doesn't matter for mean, but matters for relative importance of RCT vs External
            # Here: RCT control has weight sum N_c. External has weight sum N_samp.
            # Total weight = N_c + N_samp = N_control_final (target size).
            # This seems correct design.
            
            # Estimate ATE (Difference in Weighted Means)
            if np.sum(weights_final) > 0:
                mu_control = np.average(Y_control_final, weights=weights_final)
            else:
                mu_control = np.mean(Y_control_final) # Should not happen
                
            att_est = np.mean(Y_t) - mu_control
            
            # Compute Energy Distance between Treatment and Constructed Control (Weighted)
            # compute_energy_distance_numpy currently does not support weights
            # For Weighting method, computing unweighted energy on the full pool is misleading
            # But we can compute it on the EFFECTIVE sample if we could resample
            # For now, just compute unweighted energy on X_control_final
            # NOTE: For Weighting, X_control_final includes ALL external data. Unweighted Energy will be high (biased).
            # We should probably compute WEIGHTED energy, but our util function doesn't support it.
            # We will skip energy calculation for Weighting method to avoid misleading plots, or accept it is wrong.
            
            if method_name == "En.Weighting":
                en_w = None # Placeholder or implement weighted energy
                d12, d11, d22 = None, None, None
            else:
                en_w, d12, d11, d22 = compute_energy_distance_numpy(X_control_final, X_t)
            
            rep_results.append({
                'method': method_name,
                'n_sample': n_samp,
                'true_tau': tau,
                'att': att_est,
                'en': en_w,
                'd12': d12,
                'd11': d11,
                'd22': d22
            })
        
    return rep_results

def run_experiment():

    N_RCT = 200        # Total RCT size
    N_EXT_POOL = 1000  # External Pool size

    DIM = 2
    
    # RCT Bias is N/A since it's one population, but we define variance
    RCT_VAR = 1.0
    
    # External Bias relative to RCT
    EXT_BIAS = 1.0
    EXT_VAR = 1.5

    # Sample sizes (External samples to add)
    N_SAMPLED_LIST = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150]
    K_REP = 300
        
    print(f"Experimental Setup: N_SAMPLED={N_SAMPLED_LIST}")
    print(f"Repetitions: {K_REP}")
    print(f"Running in PARALLEL (Joblib)...")

    # Run K_REP independent simulations in parallel
    all_reps_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_repetition)(k, N_SAMPLED_LIST, N_RCT, N_EXT_POOL, DIM, 0.0, EXT_BIAS, RCT_VAR, EXT_VAR)
        for k in range(K_REP)
    )
    
    # Aggregate results
    agg_map = defaultdict(lambda: defaultdict(lambda: {
        'err': [], 'en': [], 'd12': [], 'd11': [], 'd22': []
    }))
    
    for rep_res in all_reps_results:
        for res in rep_res:
            m = res['method']
            n = res['n_sample']
            tau = res['true_tau']
            agg_map[m][n]['err'].append(res['att'] - tau)
            agg_map[m][n]['en'].append(res['en'])
            agg_map[m][n]['d12'].append(res['d12'])
            agg_map[m][n]['d11'].append(res['d11'])
            agg_map[m][n]['d22'].append(res['d22'])
    
    # Containers for final stats
    results_dict = defaultdict(list)
    raw_errors_dict = defaultdict(dict)
    
    print("-" * 30)
    print("Aggregating Results...")

    for method_name, method_data in agg_map.items():
        print(f"\nProcessing Method: {method_name}")
        for n_samp in N_SAMPLED_LIST:
            data_n = method_data[n_samp]
            
            errors = data_n['err']
            # Filter None energies (if any)
            energies_raw = data_n['en']
            energies_valid = [e for e in energies_raw if e is not None]
            
            d12_valid = [d for d in data_n['d12'] if d is not None]
            d11_valid = [d for d in data_n['d11'] if d is not None]
            d22_valid = [d for d in data_n['d22'] if d is not None]
            
            raw_errors_dict[method_name][n_samp] = errors

            bias = np.mean(errors)
            var = np.var(errors)
            
            if energies_valid:
                mean_en = np.mean(energies_valid)
                std_en = np.std(energies_valid)
                mean_d12 = np.mean(d12_valid)
                mean_d11 = np.mean(d11_valid)
                mean_d22 = np.mean(d22_valid)
            else:
                mean_en = np.nan
                std_en = np.nan
                mean_d12 = np.nan
                mean_d11 = np.nan
                mean_d22 = np.nan
            
            results_dict[method_name].append({
                'n_sample': n_samp,
                'bias': bias,
                'variance': var,
                'mean_energy': mean_en,
                'std_energy': std_en,
                'mean_d12': mean_d12,
                'mean_d11': mean_d11,
                'mean_d22': mean_d22
            })
            
            print(f"  N={n_samp}: Bias={bias:.3f} Var={var:.3f}")

    print("-" * 30)
    print("Generating Comparison Plots...")
    
    plot_bias_variance_comparison(results_dict)
    plot_energy_comparison(results_dict)
    plot_energy_mse_method_decomposition(results_dict)
    plot_error_boxplot(raw_errors_dict)
    
    # --- Detailed Plots for One Run ---
    print("Generating detailed density plots for a single example run...")
    tau = generate_tau()
    
    mean_rct = np.ones(DIM)
    X_rct = generate_covariates(N_RCT, DIM, mean_rct, var=RCT_VAR)
    mean_ext = mean_rct - EXT_BIAS
    X_ext = generate_covariates(N_EXT_POOL, DIM, mean_ext, var=EXT_VAR)
    
    beta = np.ones(DIM)
    Y0_rct, Y1_rct = generate_outcomes_nonlinear(X_rct, tau, beta=beta)
    Y0_ext, _ = generate_outcomes_nonlinear(X_ext, tau, beta=beta)
    
    last_n = N_SAMPLED_LIST[-1] # Largest sample
    n_treated = int((N_RCT + last_n) / 2)
    
    perm = np.random.permutation(N_RCT)
    idx_t = perm[:n_treated]
    idx_c = perm[n_treated:]
    
    X_t = X_rct[idx_t]
    Y_t = Y1_rct[idx_t]
    X_c_rct = X_rct[idx_c]
    Y_c_rct = Y0_rct[idx_c]
    
    for method_name, (MethodClass, kwargs) in METHODS_CONFIG.items():
        print(f"Plotting densities for {method_name}...")
        
        # Fit
        X_dummy_ctrl = np.empty((0, DIM))
        Y_dummy_ctrl = np.empty((0,))
        
        augmenter = MethodClass(n_sampled=last_n, lr=0.01, n_iter=200, **kwargs)
        augmenter.fit(X_rct, X_dummy_ctrl, X_ext)
        X_match, Y_match, weights_match = augmenter.sample(X_rct, X_dummy_ctrl, X_ext, Y_dummy_ctrl, Y0_ext)
        
        X_control_final = np.vstack([X_c_rct, X_match])
        Y_control_final = np.concatenate([Y_c_rct, Y_match])
        
        X_dict_final = {
            "RCT Treatment": X_t,
            "RCT Control (Initial)": X_c_rct,
            "Matched External": X_match,
            "Full Control (Combined)": X_control_final,
            "External Pool": X_ext
        }
        
        Y_dict_final = {
            "RCT Treatment": Y_t,
            "RCT Control (Initial)": Y_c_rct,
            "Matched External": Y_match,
            "Full Control (Combined)": Y_control_final,
            "External Pool": Y0_ext
        }
        
        plot_covariate_densities(X_dict_final, DIM, output_dir=f"plots/{method_name}")
        plot_outcome_densities(Y_dict_final, output_dir=f"plots/{method_name}")
        if DIM >= 2:
            plot_covariate_2d_scatter(X_dict_final, output_dir=f"plots/{method_name}")

if __name__ == "__main__":
    run_experiment()