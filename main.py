import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from collections import defaultdict
import os

from data_gen import create_complex_dataset
from method_whole import NaivePooled, NaiveRCT, WholeIPW, EnergyRouter

# --- Configuration ---
METHODS = {
    "Naive Pooled": NaivePooled,
    "Naive RCT": NaiveRCT,
    "Whole IPW": WholeIPW,
    "Energy Router": EnergyRouter
}

def generate_tau():
    return 2.5

def process_repetition(rep_id, n_treat, n_ctrl, n_ext, dim, rct_bias, ext_bias, rct_var, ext_var):
    tau = generate_tau()
    data = create_complex_dataset(n_treat, n_ctrl, n_ext, dim, tau, rct_bias=rct_bias, ext_bias=ext_bias, rct_var=rct_var, ext_var=ext_var)

    X_t = data["target"]["X"]
    Y_t = data["target"]["Y"]
    X_i = data["internal"]["X"]
    Y_i = data["internal"]["Y"]
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    
    res = {}
    
    for name, MethodClass in METHODS.items():
        method = MethodClass()
        try:
            att, n_obs = method.estimate(X_t, Y_t, X_i, Y_i, X_e, Y_e)
            res[name] = {
                'att': att,
                'error': att - tau,
                'n_obs': n_obs
            }
        except Exception as e:
            print(f"Error in {name}: {e}")
            res[name] = {
                'att': np.nan,
                'error': np.nan,
                'n_obs': np.nan
            }
            
    return res

def plot_results(results_map, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Prepare data for boxplot
    labels = []
    data = []
    
    sorted_names = sorted(results_map.keys())
    
    print("\n--- Summary Statistics ---")
    print(f"{'Method':<15} | {'Bias':<10} | {'Variance':<10} | {'MSE':<10} | {'RMSE':<10}")
    print("-" * 65)
    
    for name in sorted_names:
        errors = results_map[name]['errors']
        # Remove NaNs
        errors = [e for e in errors if not np.isnan(e)]
        
        if not errors:
            continue
            
        bias = np.mean(errors)
        var = np.var(errors)
        mse = np.mean(np.array(errors)**2)
        rmse = np.sqrt(mse)
        
        print(f"{name:<15} | {bias:<10.4f} | {var:<10.4f} | {mse:<10.4f} | {rmse:<10.4f}")
        names_label = name + f" (n={np.mean(results_map[name]['n_obs']):.1f})"
        labels.append(names_label)
        data.append(errors)
        
    # Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.axhline(0, color='r', linestyle='--', alpha=0.7)
    plt.title("Estimation Error Distribution by Method")
    plt.ylabel("Error (Estimate - True)")
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(output_dir, "whole_methods_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nPlot saved to {out_path}")

def run_experiment():
    N_TREAT = 150
    N_CTRL_RCT = 50
    N_EXT_POOL = 1000

    DIM = 3
    
    RCT_BIAS = 0.
    EXT_BIAS = 1.0
    RCT_VAR = 1.0
    EXT_VAR = 2.0
    
    K_REP = 100
    
    print(f"Running {K_REP} repetitions...")
    
    results_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_repetition)(k, N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, RCT_BIAS, EXT_BIAS, RCT_VAR, EXT_VAR)
        for k in range(K_REP)
    )
    
    # Aggregate
    aggregated = defaultdict(lambda: {'atts': [], 'errors': [], 'n_obs': []})
    
    for res in results_list:
        for name, metrics in res.items():
            aggregated[name]['atts'].append(metrics['att'])
            aggregated[name]['errors'].append(metrics['error'])
            aggregated[name]['n_obs'].append(metrics['n_obs'])
            
    plot_results(aggregated)

if __name__ == "__main__":
    run_experiment()
