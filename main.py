import numpy as np
from data_gen import create_complex_dataset
from methods import EnergyAugmenter
from utils import compute_energy_distance_numpy, calculate_bias_rmse
from visualization import plot_covariate_densities, plot_outcome_densities, plot_covariate_2d_scatter
import torch

def run_experiment():
    
    # Settings
    N_TREAT = 200
    N_CTRL_RCT = 70
    N_EXT_POOL = 1000
    N_AUGMENT = 5
    DIM = 2
    
    print(f"Generating Data (Dim={DIM})...")
    data = create_complex_dataset(N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, rct_bias=0, ext_bias=1.5)

    X_t = data["target"]["X"]
    Y_t = data["target"]["Y"]
    X_i = data["internal"]["X"]
    Y_i = data["internal"]["Y"]
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    TAU = data["tau"]
    
    print("Visualizing Initial Distributions...")
    X_dict = {
        "RCT Treatment (Target)": X_t,
        "RCT Control (Internal)": X_i,
        "External (Source)": X_e
    }
    plot_covariate_densities(X_dict, DIM)
    
    Y_dict = {
        "RCT Treatment": Y_t,
        "RCT Control": Y_i,
        "External Control": Y_e
    }
    plot_outcome_densities(Y_dict)

    print("-" * 30)
    print("Baseline: Internal Control Only")
    att_rct = np.mean(Y_t) - np.mean(Y_i)
    print(f"ATT (RCT): {att_rct:.3f} --> error: {att_rct - TAU:.3f}")
    print(f"Energy (Int vs Tgt): {compute_energy_distance_numpy(X_i, X_t):.4f}")
    
    print("-" * 30)
    print("Baseline: Naive Pooling (All External)")
    Y_pool = np.concatenate([Y_i, Y_e])
    X_pool = np.vstack([X_i, X_e])
    att_naive = np.mean(Y_t) - np.mean(Y_pool)
    print(f"ATT (Naive): {att_naive:.3f} --> error: {att_naive - TAU:.3f}")
    print(f"Energy (Pool vs Tgt): {compute_energy_distance_numpy(X_pool, X_t):.4f}")

    print("-" * 30)
    print("Method: Energy Augmentation (Sampling w/o Replacement)")

    augmenter = EnergyAugmenter(n_augment = N_AUGMENT, k_best=200)

    # 1. Learn Weights (Blind to Outcome)
    print("\tOptimizing weights...")
    augmenter.fit(X_t, X_i, X_e)
    
    # 2. Sample Cohort (Blind to Outcome)
    print("\tSampling best cohort...")
    # New signature: sample(X_t, X_i, X_e, Y_i, Y_e) -> returns full cohort
    X_final_control, Y_final_control = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
    
    # 3. Final Analysis
    # (Variables X_final_control, Y_final_control are now directly returned)
    
    att_aug = np.mean(Y_t) - np.mean(Y_final_control)
    final_dist = compute_energy_distance_numpy(X_final_control, X_t)

    print(f"ATT (Augmented): {att_aug:.3f} --> error: {att_aug - TAU:.3f}")
    print(f"Energy (Final vs Tgt): {final_dist:.4f}")

    print("-" * 30)
    print("Final Visualizations...")
    
    # Update dictionaries with Matched Sample
    X_dict_final = {
        "RCT Treatment": X_t,
        "RCT Control": X_i,
        "External": X_e,
        "Matched Sample": X_final_control
    }
    
    Y_dict_final = {
        "RCT Treatment": Y_t,
        "RCT Control": Y_i,
        "External": Y_e,
        "Matched Sample": Y_final_control
    }
    
    plot_covariate_densities(X_dict_final, DIM)
    plot_outcome_densities(Y_dict_final)
    plot_covariate_2d_scatter(X_dict_final)

if __name__ == "__main__":
    run_experiment()