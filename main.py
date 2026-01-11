import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from data_gen import create_complex_dataset
from methods import EnergyAugmenter
from utils import compute_energy_distance_numpy
from visualization import (
    plot_bias_variance_comparison,
    plot_energy_comparison,
    plot_mse_decomposition_comparison,
    plot_covariate_densities, 
    plot_outcome_densities, 
    plot_covariate_2d_scatter,
    plot_att_boxplot
)

def process_repetition(rep_id, n_sampled_list, n_treat, n_ctrl, n_ext, dim):
    """
    Runs a single simulation repetition:
    1. Generates a fresh dataset.
    2. Fits the EnergyAugmenter (once for this dataset).
    3. Loops through all requested sample sizes to compute ATT and Energy.
    
    Returns a list of result dictionaries for each n_sampled.
    """
    # 1. Generate NEW Data
    data = create_complex_dataset(n_treat, n_ctrl, n_ext, dim, rct_bias=0., ext_bias=1.)

    X_t = data["target"]["X"]
    X_i = data["internal"]["X"]
    X_e = data["external"]["X"]
    Y_t = data["target"]["Y"]
    Y_i = data["internal"]["Y"]
    Y_e = data["external"]["Y"]
    
    # 2. Fit Model
    # Optimize weights for the pool (Internal + External) to match Target
    augmenter = EnergyAugmenter(lr=0.01, n_iter=200) 
    augmenter.fit(X_t, X_i, X_e)
    
    rep_results = []
    
    for n_samp in n_sampled_list:
        # 3. Sample Weighted
        X_w, Y_w = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e, strategy='weighted', n_sampled=n_samp)
        att_w = np.mean(Y_t) - np.mean(Y_w)
        en_w = compute_energy_distance_numpy(X_w, X_t)
        
        # 4. Sample Top-N
        X_t_samp, Y_t_samp = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e, strategy='top_n', n_sampled=n_samp)
        att_t = np.mean(Y_t) - np.mean(Y_t_samp)
        en_t = compute_energy_distance_numpy(X_t_samp, X_t)
        
        # 5. Sample Weighted Hybrid
        X_wh, Y_wh = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e, strategy='weighted_hybrid', n_sampled=n_samp)
        att_wh = np.mean(Y_t) - np.mean(Y_wh)
        en_wh = compute_energy_distance_numpy(X_wh, X_t)
        
        # 6. Sample Top-N Hybrid
        X_th, Y_th = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e, strategy='top_n_hybrid', n_sampled=n_samp)
        att_th = np.mean(Y_t) - np.mean(Y_th)
        en_th = compute_energy_distance_numpy(X_th, X_t)
        
        rep_results.append({
            'n_sample': n_samp,
            'w_att': att_w,
            'w_en': en_w,
            't_att': att_t,
            't_en': en_t,
            'wh_att': att_wh,
            'wh_en': en_wh,
            'th_att': att_th,
            'th_en': en_th
        })
        
    return rep_results

def run_experiment():
    # Settings
    N_TREAT = 1000
    N_CTRL_RCT = 50
    N_EXT_POOL = 750
    DIM = 4
    
    # Simulation Parameters
    # Loop over sample sizes
    N_SAMPLED_LIST = np.arange(20, 155, 10).tolist()
    
    # Number of repetitions
    K_REP = 100 
    
    # Check one generation to get TAU
    temp_data = create_complex_dataset(N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM)
    TAU = temp_data["tau"]
    
    print(f"Experimental Setup: N_SAMPLED={N_SAMPLED_LIST}")
    print(f"Repetitions: {K_REP}")
    print(f"True TAU: {TAU}")
    print(f"Running in PARALLEL (Joblib)...")

    # Run K_REP independent simulations in parallel
    # n_jobs=-1 uses all available cores
    all_reps_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_repetition)(k, N_SAMPLED_LIST, N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM)
        for k in range(K_REP)
    )
    
    # Aggregate results
    # Structure: n_sample -> list of values
    agg_map = defaultdict(lambda: {
        'w_att': [], 'w_en': [], 
        't_att': [], 't_en': [],
        'wh_att': [], 'wh_en': [],
        'th_att': [], 'th_en': []
    })
    
    for rep_res in all_reps_results:
        for res in rep_res:
            n = res['n_sample']
            agg_map[n]['w_att'].append(res['w_att'])
            agg_map[n]['w_en'].append(res['w_en'])
            agg_map[n]['t_att'].append(res['t_att'])
            agg_map[n]['t_en'].append(res['t_en'])
            agg_map[n]['wh_att'].append(res['wh_att'])
            agg_map[n]['wh_en'].append(res['wh_en'])
            agg_map[n]['th_att'].append(res['th_att'])
            agg_map[n]['th_en'].append(res['th_en'])

    # Compute Statistics and format for plotting
    
    # Containers for final stats
    methods_keys = ['w', 't', 'wh', 'th']
    methods_names = {
        'w': 'Weighted',
        't': 'Top-N',
        'wh': 'Weighted Hybrid',
        'th': 'Top-N Hybrid'
    }
    
    results_dict = {name: [] for name in methods_names.values()}
    raw_atts_dict = {name: {} for name in methods_names.values()}
    
    print("-" * 30)
    print("Aggregating Results...")

    for n_samp in N_SAMPLED_LIST:
        data_n = agg_map[n_samp]
        
        for key in methods_keys:
            name = methods_names[key]
            
            atts = data_n[f'{key}_att']
            energies = data_n[f'{key}_en']
            
            # Store raw ATTs for boxplot
            raw_atts_dict[name][n_samp] = atts

            # Compute Stats
            bias = np.mean(atts) - TAU
            var = np.var(atts)
            mean_en = np.mean(energies)
            std_en = np.std(energies)
            
            results_dict[name].append({
                'n_sample': n_samp,
                'bias': bias,
                'variance': var,
                'mean_energy': mean_en,
                'std_energy': std_en
            })
            
            if key == 'w': # Just print one for progress
                print(f"N={n_samp}: {name} Bias={bias:.3f}")

    print("-" * 30)
    print("Generating Comparison Plots...")
    
    plot_bias_variance_comparison(results_dict)
    plot_energy_comparison(results_dict)
    plot_mse_decomposition_comparison(results_dict)
    plot_att_boxplot(raw_atts_dict, TAU)
    
    # Generate detailed plots for ONE representative run (locally)
    print("Generating detailed density plots for a single example run...")
    data = create_complex_dataset(N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, rct_bias=0., ext_bias=0.0)
    
    X_t = data["target"]["X"]
    Y_t = data["target"]["Y"]
    X_i = data["internal"]["X"]
    Y_i = data["internal"]["Y"]
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    
    augmenter = EnergyAugmenter(lr=0.01, n_iter=200) 
    augmenter.fit(X_t, X_i, X_e)
    
    # Use the largest sample size for the density plot visualization
    last_n = N_SAMPLED_LIST[-1]
    X_w, Y_w = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e, strategy='weighted', n_sampled=last_n)
    
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