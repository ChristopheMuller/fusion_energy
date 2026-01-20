import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from data_gen import create_complex_dataset
from methods import EnergyAugmenter_PooledTarget, IPWAugmenter
from utils import compute_energy_distance_numpy, compute_weighted_energy_distance
from visualization import (
    plot_bias_variance_comparison,
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
    "En.Matching_Pooled.005": (EnergyAugmenter_PooledTarget, {'k_best': 5}),
    "En.Matching_Pooled.100": (EnergyAugmenter_PooledTarget, {'k_best': 100}),
    "IPW": (IPWAugmenter, {}),
}

def tau_fn(X):
    return np.ones(X.shape[0]) *  1.5

def process_repetition(rep_id, n_sampled_list, n_rct, n_ext, dim, rct_bias, ext_bias, rct_var, ext_var, tau):
    """
    Runs a single simulation repetition:
    1. Generates a fresh dataset (Full RCT + External).
    2. Loops through all requested sample sizes to fit and sample.
    3. Randomly splits RCT into Treated/Control depending on external sample size.
    
    Returns a list of result dictionaries for each n_sampled.
    """
    # 1. Generate NEW Data
    data = create_complex_dataset(n_rct, 0, n_ext, dim, tau, rct_bias=rct_bias, ext_bias=ext_bias, rct_var=rct_var, ext_var=ext_var)

    X_rct = data["target"]["X"]
    Y0_rct = data["target"]["Y0"]
    Y1_rct = data["target"]["Y1"]
    
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    
    # True ATT on the RCT population
    true_att = data['true_att'] 
    
    rep_results = []
    
    for n_samp in n_sampled_list:
        for method_name, (MethodClass, kwargs) in METHODS_CONFIG.items():
            # 2. Fit and Sample Method for each n_sampled
            # Target is RCT (full), Internal is Empty.
            augmenter = MethodClass(n_sampled=n_samp, lr=0.01, n_iter=200, **kwargs) 
            
            # Pass empty Internal Control
            X_empty = np.empty((0, dim))
            augmenter.fit(X_rct, X_empty, X_e)
            
            # Sample/Weight
            # Y_empty corresponds to Internal outcomes (none)
            Y_empty = np.empty((0,))
            X_fused, Y_fused, weights = augmenter.sample(X_rct, X_empty, X_e, Y_empty, Y_e)
            
            # 3. Random Split of RCT
            if method_name.startswith("IPW") or method_name.startswith("En.Weighting"):
                 # Using all external data (weighted)
                 n_chosen = X_e.shape[0]
            else:
                 # Matching methods select n_sampled
                 n_chosen = n_samp
            
            # Ensure balanced design: n_treated approx equal to total control size (internal + chosen external)
            n_treat_target = int((n_rct + n_chosen) / 2)
            
            # Ensure boundaries
            n_treat_target = max(1, min(n_rct - 1, n_treat_target))
            
            # Random shuffle indices
            indices = np.arange(n_rct)
            np.random.shuffle(indices)
            
            idx_t = indices[:n_treat_target]
            idx_c = indices[n_treat_target:]
            
            X_t_sim = X_rct[idx_t]
            Y_t_sim = Y1_rct[idx_t] # Observed Y for Treated is Y1
            
            X_c_sim = X_rct[idx_c]
            Y_c_sim = Y0_rct[idx_c] # Observed Y for Control is Y0
            
            # 4. Estimate ATT
            if "Matching" in method_name:
                Y_control_total = np.concatenate([Y_c_sim, Y_fused])
                mu_control = np.mean(Y_control_total)
                
                att_est = np.mean(Y_t_sim) - mu_control
                
                X_control_total = np.vstack([X_c_sim, X_fused])
                w_total = np.ones(len(X_control_total)) / len(X_control_total)
                
                en_w, d12, d11, d22 = compute_weighted_energy_distance(X_control_total, X_t_sim, w_total)

            else:
                
                w_ext_norm = weights # sum to 1
                w_ext_scaled = w_ext_norm * n_chosen # sum to n_chosen
                
                w_c = np.ones(len(Y_c_sim)) # weight 1 each
                
                w_total = np.concatenate([w_c, w_ext_scaled])
                w_total = w_total / np.sum(w_total) # normalize total
                
                Y_total = np.concatenate([Y_c_sim, Y_e])
                mu_control = np.sum(w_total * Y_total)
                
                att_est = np.mean(Y_t_sim) - mu_control
                
                # Energy Distance for plotting/metrics
                X_control_total = np.vstack([X_c_sim, X_e])
                en_w, d12, d11, d22 = compute_weighted_energy_distance(X_control_total, X_t_sim, w_total)

            rep_results.append({
                'method': method_name,
                'n_sample': n_samp,
                'true_tau': true_att,
                'att': att_est,
                'en': en_w,
                'd12': d12,
                'd11': d11,
                'd22': d22
            })
        
    return rep_results

def run_experiment():

    N_RCT = 200
    N_EXT_POOL = 1000

    DIM = 3
    
    RCT_BIAS = 0.
    EXT_BIAS = 1.
    RCT_VAR = 1.0
    EXT_VAR = 2.0

    TAU = tau_fn

    # Restoring full experiment settings
    N_SAMPLED_LIST = [0, 5, 10, 20, 30, 50, 75, 100, 150]
    K_REP = 150
        
    print(f"Experimental Setup: N_SAMPLED={N_SAMPLED_LIST}")
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
        'err': [], 'en': [], 'd12': [], 'd11': [], 'd22': []
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
    plot_energy_mse_method_decomposition(results_dict)
    plot_error_boxplot(raw_errors_dict)
    
    print("Detailed density plots skipped as per new setup instructions.")

if __name__ == "__main__":
    run_experiment()