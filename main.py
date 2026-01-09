import numpy as np
from data_gen import create_complex_dataset
from methods import EnergyAugmenter
from utils import compute_energy_distance_numpy, calculate_bias_rmse
from visualization import plot_covariate_densities, plot_outcome_densities
import torch

def run_experiment():
    
    # Settings
    N_TREAT = 200
    N_CTRL_RCT = 50
    N_EXT_POOL = 1000
    N_AUGMENT = 50
    DIM = 6
    
    print(f"Generating Data (Dim={DIM})...")
    data = create_complex_dataset(N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, rct_bias=0., ext_bias=1.0)

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
    X_aug, Y_aug = augmenter.sample(X_t, X_i, X_e, Y_e)
    
    # 3. Final Analysis
    Y_final_control = np.concatenate([Y_i, Y_aug])
    X_final_control = np.vstack([X_i, X_aug])
    
    att_aug = np.mean(Y_t) - np.mean(Y_final_control)
    final_dist = compute_energy_distance_numpy(X_final_control, X_t)

    print(f"ATT (Augmented): {att_aug:.3f} --> error: {att_aug - TAU:.3f}")
    print(f"Energy (Final vs Tgt): {final_dist:.4f}")

if __name__ == "__main__":
    run_experiment()