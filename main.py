import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from data_gen import create_complex_dataset
from methods import EnergyAugmenter_Weighted
from utils import compute_energy_distance_numpy
from visualization import (
    plot_bias_variance_comparison,
    plot_energy_comparison,
    plot_mse_decomposition_comparison,
    plot_covariate_densities, 
    plot_outcome_densities, 
    plot_covariate_2d_scatter,
    plot_error_boxplot,
    plot_energy_mse_method_decomposition
)


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
        # 2. Fit and Sample Weighted Method for each n_sampled
        augmenter = EnergyAugmenter_Weighted(n_sampled=n_samp, k_best=1, lr=0.01, n_iter=200) 
        augmenter.fit(X_t, X_i, X_e)
        
        X_w, Y_w = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
        
        att_w = np.mean(Y_t) - np.mean(Y_w)
        en_w = compute_energy_distance_numpy(X_w, X_t)
        
        rep_results.append({
            'n_sample': n_samp,
            'true_tau': tau,
            'att': att_w,
            'en': en_w
        })
        
    return rep_results

def run_experiment():

    N_TREAT = 1000
    N_CTRL_RCT = 50
    N_EXT_POOL = 1000
    DIM = 4

    RCT_BIAS = 0.0
    EXT_BIAS = 1.0
    RCT_VAR = 1.0
    EXT_VAR = 2.0

    # Restoring full experiment settings
    # N_SAMPLED_LIST = np.arange(20, 155, 10).tolist()
    # K_REP = 30 
    N_SAMPLED_LIST = [0, 5, 10, 15, 20, 25, 30]
    K_REP = 10
        
    print(f"Experimental Setup: N_SAMPLED={N_SAMPLED_LIST}")
    print(f"Repetitions: {K_REP}")
    print(f"Running in PARALLEL (Joblib)...")

    # Run K_REP independent simulations in parallel
    all_reps_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_repetition)(k, N_SAMPLED_LIST, N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, RCT_BIAS, EXT_BIAS, RCT_VAR, EXT_VAR)
        for k in range(K_REP)
    )
    
    # Aggregate results
    agg_map = defaultdict(lambda: {
        'err': [], 'en': []
    })
    
    for rep_res in all_reps_results:
        for res in rep_res:
            n = res['n_sample']
            tau = res['true_tau']
            agg_map[n]['err'].append(res['att'] - tau)
            agg_map[n]['en'].append(res['en'])
    
    # Containers for final stats
    method_name = "Weighted"
    results_dict = {method_name: []}
    raw_errors_dict = {method_name: {}}
    
    print("-" * 30)
    print("Aggregating Results...")

    for n_samp in N_SAMPLED_LIST:
        data_n = agg_map[n_samp]
        
        errors = data_n['err']
        energies = data_n['en']
        
        # Store raw Errors for boxplot
        raw_errors_dict[method_name][n_samp] = errors

        # Compute Stats based on ERRORS
        bias = np.mean(errors)
        var = np.var(errors)
        mean_en = np.mean(energies)
        std_en = np.std(energies)
        
        results_dict[method_name].append({
            'n_sample': n_samp,
            'bias': bias,
            'variance': var,
            'mean_energy': mean_en,
            'std_energy': std_en
        })
        
        print(f"N={n_samp}: Bias={bias:.3f} Var={var:.3f}")

    print("-" * 30)
    print("Generating Comparison Plots...")
    
    plot_bias_variance_comparison(results_dict)
    plot_energy_comparison(results_dict)
    plot_mse_decomposition_comparison(results_dict)
    plot_energy_mse_method_decomposition(results_dict)
    plot_error_boxplot(raw_errors_dict)
    
    # Generate detailed plots for ONE representative run (locally)
    print("Generating detailed density plots for a single example run...")
    tau = generate_tau()
    data = create_complex_dataset(N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, tau, rct_bias=RCT_BIAS, ext_bias=EXT_BIAS, rct_var=RCT_VAR, ext_var=EXT_VAR)
    
    X_t = data["target"]["X"]
    Y_t = data["target"]["Y"]
    X_i = data["internal"]["X"]
    Y_i = data["internal"]["Y"]
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    
    # Use the largest sample size for the density plot visualization
    last_n = N_SAMPLED_LIST[-1]
    augmenter = EnergyAugmenter_Weighted(n_sampled=last_n, k_best=50, lr=0.01, n_iter=200) 
    augmenter.fit(X_t, X_i, X_e)
    X_w, Y_w = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
    
    X_dict_final = {
        "RCT Treatment": X_t,
        "RCT Control": X_i,
        "External": X_e,
        "Matched Sample (Weighted)": X_w
    }
    Y_dict_final = {
        "RCT Treatment": Y_t,
        "RCT Control": Y_i,
        "External": Y_e,
        "Matched Sample (Weighted)": Y_w
    }
    plot_covariate_densities(X_dict_final, DIM)
    plot_outcome_densities(Y_dict_final)
    plot_covariate_2d_scatter(X_dict_final)

if __name__ == "__main__":
    run_experiment()