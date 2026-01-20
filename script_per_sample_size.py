import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from data_gen import create_complex_dataset
from methods import EnergyAugmenter_Matching, EnergyAugmenter_PooledTarget, EnergyAugmenter_MatchingReg, IPWAugmenter, EnergyAugmenter_Weighting
from utils import compute_energy_distance_numpy, compute_weighted_energy_distance
from visualization import (
    plot_bias_variance_comparison,
    plot_energy_comparison,
    plot_covariate_densities,
    plot_outcome_densities,
    plot_covariate_2d_scatter,
    plot_covariate_2d_scatter_weighted,
    plot_error_boxplot,
    plot_energy_mse_method_decomposition,
    plot_treatment_effect_heterogeneity
)

# --- Global Configuration ---
METHODS_CONFIG = {
    "En.Matching_005k": (EnergyAugmenter_Matching, {'k_best': 5}),
    "En.Matching_100k": (EnergyAugmenter_Matching, {'k_best': 100}),
    "IPW": (IPWAugmenter, {}),
    "En.Weighting": (EnergyAugmenter_Weighting, {})
}

def tau_fn(X):
    return np.ones(X.shape[0]) *  1.5

def process_repetition(rep_id, n_sampled_list, n_rct, n_ext, dim, rct_bias, ext_bias, rct_var, ext_var, tau):
    """
    Runs a single simulation repetition:
    1. Generates Pooled RCT (unbiased) and External data.
    2. Loops through n_sampled_list:
       (A) Pre-split: Measure energy between Pooled RCT and best matching External sample.
       (B) Split RCT into Treated/Control to balance sizes.
       (C) Run methods on the split data.
    """
    # 1. Generate NEW Data
    data = create_complex_dataset(n_rct, 0, n_ext, dim, tau, rct_bias=0.0, ext_bias=ext_bias, rct_var=rct_var, ext_var=ext_var)

    X_rct_pool = data["target"]["X"]
    # Y_rct_pool = data["target"]["Y"] # This is Y1 for target, we need potential outcomes
    Y0_rct_pool = data["target"]["Y0"]
    Y1_rct_pool = data["target"]["Y1"]
    
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    
    true_att = data['true_att']
    
    rep_results = []    
    indices_rct = np.arange(n_rct)
    
    for n_samp in n_sampled_list:
        
        # --- (A) Measure Pooled Energy ---
        # Minimum energy between a matching sample of size n_sampled and the pooled RCT data.
        en_pooled = None
        if n_samp > 0:
            # We use EnergyAugmenter_Matching to find the best subset of External that matches X_rct_pool
            # Target = X_rct_pool
            # Internal = Empty
            matcher_pool = EnergyAugmenter_Matching(n_sampled=n_samp, k_best=20, lr=0.01, n_iter=200)
            X_empty = np.zeros((0, dim))
            
            matcher_pool.fit(X_rct_pool, X_empty, X_e)
            # sample returns X_fused. Since Internal is empty, X_fused is just the matched external sample.
            X_matched_pool, _ = matcher_pool.sample(X_rct_pool, X_empty, X_e)
            
            # Compute Energy(Matched, Pooled RCT)
            en_pooled, _, _, _ = compute_energy_distance_numpy(X_matched_pool, X_rct_pool)
        else:
            en_pooled = 0.0 # No external sample, energy of empty? Or technically undefined. Let's set to 0 or nan.
            # If n_sampled=0, we just have RCT. The "Matching Energy" is not applicable or is 0 distance if we consider we add nothing.
            # But for plotting, let's keep it None or 0.
        
        # --- (B) Split Pooled RCT ---
        # n_treated = (n_rct + n_sampled) / 2
        n_t = int((n_rct + n_samp) / 2)
        n_c = n_rct - n_t
        
        if n_c < 2: 
            # If n_sampled is too large, n_c becomes too small. 
            # n_t <= n_rct => (n_rct + n_samp)/2 <= n_rct => n_samp <= n_rct.
            # We assume n_samp <= n_rct roughly.
            continue

        np.random.shuffle(indices_rct)
        idx_t = indices_rct[:n_t]
        idx_c = indices_rct[n_t:]
        
        X_t = X_rct_pool[idx_t]
        # Y_t should be Y1 (Treated)
        Y_t = Y1_rct_pool[idx_t]
        
        X_i = X_rct_pool[idx_c]
        # Y_i should be Y0 (Control)
        Y_i = Y0_rct_pool[idx_c]

        # --- (C) Run Methods ---
        for method_name, (MethodClass, kwargs) in METHODS_CONFIG.items():
            # 2. Fit and Sample Method for each n_sampled
            augmenter = MethodClass(n_sampled=n_samp, lr=0.01, n_iter=200, **kwargs) 
            augmenter.fit(X_t, X_i, X_e)
            
            # Updated to unpack weights
            X_w, Y_w, weights = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
            
            # Compute ATT using weighted sum estimator
            Y_control_full = np.concatenate([Y_i, Y_e])
            mu_control_weighted = np.sum(weights * Y_control_full)
            
            att_w = np.mean(Y_t) - mu_control_weighted
            
            # Unify Energy Calculation:
            X_control_full = np.vstack([X_i, X_e])
            en_w, d12, d11, d22 = compute_weighted_energy_distance(X_control_full, X_t, weights)
            
            rep_results.append({
                'method': method_name,
                'n_sample': n_samp,
                'true_tau': true_att,
                'att': att_w,
                'en': en_w,
                'd12': d12,
                'd11': d11,
                'd22': d22,
                'en_pooled': en_pooled
            })
        
    return rep_results

def run_experiment():

    N_RCT = 300
    N_EXT_POOL = 1000

    DIM = 3
    
    RCT_BIAS = 0.
    EXT_BIAS = 1.
    RCT_VAR = 1.0
    EXT_VAR = 2.0

    TAU = tau_fn

    # Restoring full experiment settings
    N_SAMPLED_LIST = [0, 5, 10, 20, 30, 50]
    K_REP = 150
        
    print(f"Experimental Setup: N_RCT={N_RCT}, N_SAMPLED={N_SAMPLED_LIST}")
    print(f"Repetitions: {K_REP}")
    print(f"Running in PARALLEL (Joblib)...")

    # Run K_REP independent simulations in parallel
    all_reps_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_repetition)(k, N_SAMPLED_LIST, N_RCT, N_EXT_POOL, DIM, RCT_BIAS, EXT_BIAS, RCT_VAR, EXT_VAR, TAU)
        for k in range(K_REP)
    )
    
    # Aggregate results
    # Structure: agg_map[method][n_sample] = {'err': [], 'en': [], ...}
    agg_map = defaultdict(lambda: defaultdict(lambda: {
        'err': [], 'en': [], 'd12': [], 'd11': [], 'd22': [], 'en_pooled': []
    }))
    
    for rep_res in all_reps_results:
        for res in rep_res:
            m = res['method']
            n = res['n_sample']
            true_tau = res['true_tau']
            agg_map[m][n]['err'].append(res['att'] - true_tau)
            agg_map[m][n]['en'].append(res['en'])
            agg_map[m][n]['d12'].append(res['d12'])
            agg_map[m][n]['d11'].append(res['d11'])
            agg_map[m][n]['d22'].append(res['d22'])
            agg_map[m][n]['en_pooled'].append(res['en_pooled'])
    
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
            # Filter None energies
            energies_raw = data_n['en']
            energies_valid = [e for e in energies_raw if e is not None]
            
            d12_valid = [d for d in data_n['d12'] if d is not None]
            d11_valid = [d for d in data_n['d11'] if d is not None]
            d22_valid = [d for d in data_n['d22'] if d is not None]

            en_pooled_raw = data_n['en_pooled']
            en_pooled_valid = [e for e in en_pooled_raw if e is not None]
            
            # Store raw Errors for boxplot
            raw_errors_dict[method_name][n_samp] = errors

            # Compute Stats based on ERRORS
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
            
            if en_pooled_valid:
                mean_en_pooled = np.mean(en_pooled_valid)
            else:
                mean_en_pooled = np.nan

            results_dict[method_name].append({
                'n_sample': n_samp,
                'bias': bias,
                'variance': var,
                'mean_energy': mean_en,
                'std_energy': std_en,
                'mean_d12': mean_d12,
                'mean_d11': mean_d11,
                'mean_d22': mean_d22,
                'mean_pooled_energy': mean_en_pooled
            })
            
            print(f"  N={n_samp}: Bias={bias:.3f} Var={var:.3f}")

    print("-" * 30)
    print("Generating Comparison Plots...")
    
    plot_bias_variance_comparison(results_dict)
    plot_energy_comparison(results_dict)
    plot_energy_mse_method_decomposition(results_dict)
    plot_error_boxplot(raw_errors_dict)
    
    # Determine best N (min MSE) for each method
    best_n_per_method = {}
    print("\nBest Sample Sizes per Method (min MSE):")
    for method_name, res_list in results_dict.items():
        # entries have 'bias', 'variance', 'n_sample'
        best_entry = min(res_list, key=lambda x: x['bias']**2 + x['variance'])
        best_n_per_method[method_name] = best_entry['n_sample']
        mse = best_entry['bias']**2 + best_entry['variance']
        print(f"  {method_name}: N={best_entry['n_sample']} (MSE={mse:.4f})")

    # Generate detailed plots for ONE representative run (locally) but for ALL methods
    # We need to manually replicate the split logic for this single run
    print("Generating detailed density plots for a single example run (all methods)...")
    
    # Generate Pooled RCT
    data = create_complex_dataset(N_RCT, 0, N_EXT_POOL, DIM, TAU, rct_bias=0.0, ext_bias=EXT_BIAS, rct_var=RCT_VAR, ext_var=EXT_VAR)
    X_rct_pool = data["target"]["X"]
    # Y_rct_pool = data["target"]["Y"]
    Y0_rct_pool = data["target"]["Y0"]
    Y1_rct_pool = data["target"]["Y1"]

    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    
    # Check heterogeneity
    if callable(TAU):
        print("Plotting treatment effect heterogeneity...")
        tau_vals = TAU(X_rct_pool)
        plot_treatment_effect_heterogeneity(X_rct_pool, tau_vals, output_dir="plots")
        
    for method_name, (MethodClass, kwargs) in METHODS_CONFIG.items():
        best_n = best_n_per_method.get(method_name, N_SAMPLED_LIST[-1])
        
        # Re-Split based on best_n
        n_t = int((N_RCT + best_n) / 2)
        indices_rct = np.arange(N_RCT)
        np.random.shuffle(indices_rct)
        idx_t = indices_rct[:n_t]
        idx_c = indices_rct[n_t:]
        
        X_t = X_rct_pool[idx_t]
        # Y_t = Y_rct_pool[idx_t]
        Y_t = Y1_rct_pool[idx_t]

        X_i = X_rct_pool[idx_c]
        # Y_i = Y_rct_pool[idx_c]
        Y_i = Y0_rct_pool[idx_c]

        print(f"Plotting densities for {method_name} (using Best N={best_n})...")
        augmenter = MethodClass(n_sampled=best_n, lr=0.01, n_iter=200, **kwargs) 
        augmenter.fit(X_t, X_i, X_e)
        X_w, Y_w, _ = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
        internal_weights = augmenter.get_internal_weights()
        
        X_dict_final = {
            "RCT Treatment": X_t,
            "RCT Control": X_i,
            "External": X_e
        }
        Y_dict_final = {
            "RCT Treatment": Y_t,
            "RCT Control": Y_i,
            "External": Y_e
        }
        
        # Only add Matched Sample if it exists (i.e. not IPW or weighting method that returns None)
        if X_w is not None and Y_w is not None:
            X_matched = X_w
            Y_matched = Y_w
            X_dict_final[f"Fused Control ({method_name})"] = X_matched
            Y_dict_final[f"Fused Control ({method_name})"] = Y_w
        
        # Save in method specific folder
        plot_covariate_densities(X_dict_final, DIM, output_dir=f"plots/{method_name}")
        plot_outcome_densities(Y_dict_final, output_dir=f"plots/{method_name}")
        if DIM >= 2:
            plot_covariate_2d_scatter(X_dict_final, output_dir=f"plots/{method_name}")
            plot_covariate_2d_scatter_weighted(X_dict_final, weights=internal_weights, output_dir=f"plots/{method_name}")

if __name__ == "__main__":
    run_experiment()