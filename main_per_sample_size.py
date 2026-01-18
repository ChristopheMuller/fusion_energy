import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from data_gen import create_complex_dataset
from methods import EnergyAugmenter_Matching, EnergyAugmenter_PooledTarget, EnergyAugmenter_MatchingReg, IPWAugmenter, EnergyAugmenter_Weighting
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
METHODS_CONFIG = {
    "En.Matching (Top 1)": (EnergyAugmenter_Matching, {'k_best': 1}),
    "En.Matching (Top 100)": (EnergyAugmenter_Matching, { 'k_best': 100}),
    # "En.PooledTarget_100k": (EnergyAugmenter_PooledTarget, {'k_best': 100}),
    "IPW": (IPWAugmenter, {}),
    "En.Weighting": (EnergyAugmenter_Weighting, {})
}

def generate_tau():
    return 2.5

def process_repetition(rep_id, n_sampled_list, n_treat, n_ctrl, n_ext, dim, rct_bias, ext_bias, rct_var, ext_var):
    """
    Runs a single simulation repetition:
    1. Generates a fresh dataset.
    2. Loops through all requested sample sizes to fit and sample.
    
    Returns a list of result dictionaries for each n_sampled.
    """
    # 1. Generate NEW Data
    tau = generate_tau()
    data = create_complex_dataset(n_treat, n_ctrl, n_ext, dim, tau, rct_bias=rct_bias, ext_bias=ext_bias, rct_var=rct_var, ext_var=ext_var)

    X_t = data["target"]["X"]
    X_i = data["internal"]["X"]
    X_e = data["external"]["X"]
    Y_t = data["target"]["Y"]
    Y_i = data["internal"]["Y"]
    Y_e = data["external"]["Y"]
    
    rep_results = []
    
    for n_samp in n_sampled_list:
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
            # Handle Energy Calculation (might be None for IPW)
            en_w, d12, d11, d22 = compute_energy_distance_numpy(X_w, X_t)
            
            rep_results.append({
                'method': method_name,
                'n_sample': n_samp,
                'true_tau': tau,
                'att': att_w,
                'en': en_w,
                'd12': d12,
                'd11': d11,
                'd22': d22
            })
        
    return rep_results

def run_experiment():

    N_TREAT = 150
    N_CTRL_RCT = 150
    N_EXT_POOL = 1000

    DIM = 5

    RCT_BIAS = 0.5
    EXT_BIAS = 1.
    RCT_VAR = 1.0
    EXT_VAR = 2.0

    # Restoring full experiment settings
    N_SAMPLED_LIST = [0, 5, 10, 20, 30, 50, 75, 100, 150]
    K_REP = 100
        
    print(f"Experimental Setup: N_SAMPLED={N_SAMPLED_LIST}")
    print(f"Repetitions: {K_REP}")
    print(f"Running in PARALLEL (Joblib)...")

    # Run K_REP independent simulations in parallel
    all_reps_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_repetition)(k, N_SAMPLED_LIST, N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, RCT_BIAS, EXT_BIAS, RCT_VAR, EXT_VAR)
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
    plot_energy_comparison(results_dict)
    plot_energy_mse_method_decomposition(results_dict)
    plot_error_boxplot(raw_errors_dict)
    
    # Generate detailed plots for ONE representative run (locally) but for ALL methods
    print("Generating detailed density plots for a single example run (all methods)...")
    tau = generate_tau()
    data = create_complex_dataset(N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, tau, rct_bias=RCT_BIAS, ext_bias=EXT_BIAS, rct_var=RCT_VAR, ext_var=EXT_VAR)
    
    X_t = data["target"]["X"]
    Y_t = data["target"]["Y"]
    X_i = data["internal"]["X"]
    Y_i = data["internal"]["Y"]
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    
    last_n = N_SAMPLED_LIST[-1]
    
    for method_name, (MethodClass, kwargs) in METHODS_CONFIG.items():
        print(f"Plotting densities for {method_name}...")
        augmenter = MethodClass(n_sampled=last_n, lr=0.01, n_iter=200, **kwargs) 
        augmenter.fit(X_t, X_i, X_e)
        X_w, Y_w, _ = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
        
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
            X_dict_final[f"Matched ({method_name})"] = X_w
            Y_dict_final[f"Matched ({method_name})"] = Y_w
        
        # Save in method specific folder
        plot_covariate_densities(X_dict_final, DIM, output_dir=f"plots/{method_name}")
        plot_outcome_densities(Y_dict_final, output_dir=f"plots/{method_name}")
        if DIM >= 2:
            plot_covariate_2d_scatter(X_dict_final, output_dir=f"plots/{method_name}")

if __name__ == "__main__":
    run_experiment()