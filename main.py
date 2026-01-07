import numpy as np
from data_gen import create_dataset
from methods import optimize_weights, refined_sampling, inverse_propensity_weighting
from utils import calculate_att_error
from visualization import plot_covariate_and_outcome_distributions, plot_matched_outcomes

def run_simulation():
    np.random.seed(12345)
    
    N_RCT_TREAT = 200
    N_RCT_CONTROL = 500
    N_EXT = 5000
    K = 1000
    N_SELECT = 500
    DIM = 5
    SHIFT_EXT = 1.5 
    BETA = np.random.uniform(-1, 1, DIM)
    TAU = 2.0 
    
    for shift_type in ["linear", "quadratic"]:
        print(f"\n==========================================")
        print(f"Running Simulation with Shift Type: {shift_type.upper()}")
        print(f"==========================================")
        
        data = create_dataset(
            N_RCT_TREAT, N_RCT_CONTROL, N_EXT, DIM, SHIFT_EXT, BETA, TAU, shift_type=shift_type
        )
        
        plot_covariate_and_outcome_distributions(data, save_path=f"covariate_dist_{shift_type}.png")
        
        X_target = data["rct_treat"]["X"]
        Y_target = data["rct_treat"]["Y"]
        
        X_rct_0 = data["rct_control"]["X"]
        Y_rct_0 = data["rct_control"]["Y"]
        
        X_ext = data["external"]["X"]
        Y_ext = data["external"]["Y"]
        
        X_pool = np.vstack([X_rct_0, X_ext])
        Y_pool = np.concatenate([Y_rct_0, Y_ext])
        
        print(f"Target Size: {len(X_target)}")
        print(f"Internal Control Size: {len(X_rct_0)}")
        print(f"External Control Size: {len(X_ext)}")
        
        print("\n--- Method 1: RCT Only (Baseline) ---")
        y0_hat_rct = np.mean(Y_rct_0)
        att_rct = np.mean(Y_target) - y0_hat_rct
        print(f"RCT Only ATT: {att_rct:.3f} (Error: {calculate_att_error(att_rct, TAU):.3f})")

        print("\n--- Method 2: Naive Pooling ---")
        y0_hat_naive = np.mean(Y_pool)
        att_naive = np.mean(Y_target) - y0_hat_naive
        print(f"Naive ATT: {att_naive:.3f} (Error: {calculate_att_error(att_naive, TAU):.3f})")
        
        print("\n--- Method 3: Energy Weighted Pooling ---")
        weights = optimize_weights(X_pool, X_target)
        y0_hat_weighted = np.average(Y_pool, weights=weights)
        att_weighted = np.mean(Y_target) - y0_hat_weighted
        print(f"Weighted ATT: {att_weighted:.3f} (Error: {calculate_att_error(att_weighted, TAU):.3f})")
        
        print("\n--- Method 4: Energy Weighted Refined Sampling (Best-of-K) ---")
        X_sample, Y_sample = refined_sampling(
            X_pool, Y_pool, weights, X_target, n_select=N_SELECT, K=K
        )
        
        plot_matched_outcomes(data, Y_sample, save_path=f"matched_outcomes_{shift_type}.png")
        
        y0_hat_sample = np.mean(Y_sample)
        att_sample = np.mean(Y_target) - y0_hat_sample
        print(f"Refined Sample ATT: {att_sample:.3f} (Error: {calculate_att_error(att_sample, TAU):.3f})")

        print("\n--- Method 5: Inverse Propensity Weighting (IPW) ---")
        weights_ipw = inverse_propensity_weighting(X_pool, X_target)
        y0_hat_ipw = np.average(Y_pool, weights=weights_ipw)
        att_ipw = np.mean(Y_target) - y0_hat_ipw
        print(f"IPW ATT: {att_ipw:.3f} (Error: {calculate_att_error(att_ipw, TAU):.3f})")

if __name__ == "__main__":
    run_simulation()