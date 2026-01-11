import numpy as np
from data_gen import create_complex_dataset
from methods import EnergyAugmenter
from utils import compute_energy_distance_numpy
from visualization import (
    plot_covariate_densities, 
    plot_outcome_densities, 
    plot_covariate_2d_scatter,
    plot_simulation_boxplots,
    plot_mse_evolution,
    plot_mse_and_energy_evolution
)

def run_experiment():
    
    # Settings
    N_TREAT = 200
    N_CTRL_RCT = 100
    N_EXT_POOL = 750
    
    # Simulation Parameters
    N_SAMPLED_LIST = [20, 40, 60, 70, 80, 100, 120, 140, 160]
    K_REP = 10
    
    DIM = 4
    
    print(f"Generating Data (Dim={DIM})...")
    # Generate ONCE to isolate sampling variance
    data = create_complex_dataset(N_TREAT, N_CTRL_RCT, N_EXT_POOL, DIM, rct_bias=0., ext_bias=10.0)

    X_t = data["target"]["X"]
    Y_t = data["target"]["Y"]
    X_i = data["internal"]["X"]
    Y_i = data["internal"]["Y"]
    X_e = data["external"]["X"]
    Y_e = data["external"]["Y"]
    TAU = data["tau"]
    
    # Check baseline stats
    att_rct = np.mean(Y_t) - np.mean(Y_i)
    print(f"True TAU: {TAU}")
    print(f"Baseline RCT ATT: {att_rct:.3f} (Error: {att_rct - TAU:.3f})")

    simulation_results = []
    
    print(f"Starting Simulation over N_SAMPLED={N_SAMPLED_LIST} with K_REP={K_REP}...")
    
    for n_samp in N_SAMPLED_LIST:
        print(f"Processing N_SAMPLED = {n_samp}...")
        
        # Lists to store metrics for this n_samp
        energies = []
        atts = []
        
        augmenter = EnergyAugmenter(n_sampled=n_samp, k_best=100, n_iter=300)
        augmenter.fit(X_t, X_i, X_e)
        
        for k in range(K_REP):
            # Sample
            X_final_control, Y_final_control = augmenter.sample(X_t, X_i, X_e, Y_i, Y_e)
            
            # Compute Metrics
            # ATT
            att_est = np.mean(Y_t) - np.mean(Y_final_control)
            atts.append(att_est)
            
            # Energy
            en_dist = compute_energy_distance_numpy(X_final_control, X_t)
            energies.append(en_dist)
        
        # Store results
        res = {
            'n_sample': n_samp,
            'energies': energies,
            'atts': atts,
            'true_tau': TAU
        }
        simulation_results.append(res)
        
        # Print Stats
        mean_att = np.mean(atts)
        std_att = np.std(atts)
        mean_en = np.mean(energies)
        print(f"  -> Mean ATT: {mean_att:.3f} (Std: {std_att:.3f})")
        print(f"  -> Mean Energy: {mean_en:.4f}")

    print("-" * 30)
    print("Generating Simulation Plots...")
    plot_simulation_boxplots(simulation_results)
    plot_mse_evolution(simulation_results)
    plot_mse_and_energy_evolution(simulation_results)
    
    # Generate the single-shot visualization for the LAST configuration (largest N)
    # just to keep the old visual outputs as well
    print("Generating detailed plots for the last configuration...")
    X_dict_final = {
        "RCT Treatment": X_t,
        "RCT Control": X_i,
        "External": X_e,
        "Matched Sample (Last Run)": X_final_control # From last loop iteration
    }
    Y_dict_final = {
        "RCT Treatment": Y_t,
        "RCT Control": Y_i,
        "External": Y_e,
        "Matched Sample (Last Run)": Y_final_control
    }
    plot_covariate_densities(X_dict_final, DIM)
    plot_outcome_densities(Y_dict_final)
    plot_covariate_2d_scatter(X_dict_final)

if __name__ == "__main__":
    run_experiment()
