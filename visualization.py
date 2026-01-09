import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA

def plot_covariate_densities(data_dict, dim, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for d in range(dim):
        plt.figure(figsize=(10, 6))
        for label, X in data_dict.items():
            # Adjust style for Matched Sample if present to make it stand out
            if "Matched" in label:
                plt.hist(X[:, d], bins=30, density=True, alpha=0.3, label=label, 
                         histtype='step', linewidth=2, color='black', linestyle='--')
            else:
                plt.hist(X[:, d], bins=30, density=True, alpha=0.5, label=label, histtype='stepfilled')
        
        plt.title(f"Density Comparison: Covariate {d+1}")
        plt.xlabel(f"Value of Covariate {d+1}")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/covariate_{d+1}_density.png")
        plt.close()
    print(f"Covariate plots saved to {output_dir}/")

def plot_outcome_densities(outcome_dict, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    for label, Y in outcome_dict.items():
        if "Matched" in label:
            plt.hist(Y, bins=30, density=True, alpha=0.3, label=label, 
                     histtype='step', linewidth=2, color='black', linestyle='--')
        else:
            plt.hist(Y, bins=30, density=True, alpha=0.5, label=label, histtype='stepfilled')
    
    plt.title("Density Comparison: Outcomes")
    plt.xlabel("Outcome Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/outcome_density.png")
    plt.close()
    print(f"Outcome plot saved to {output_dir}/")

def plot_covariate_2d_scatter(data_dict, output_dir="plots"):
    """
    Scatter plot of the first 2 Principal Components (PCA).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    
    # Concatenate all data to fit PCA on the global distribution space
    all_data_list = [X for X in data_dict.values() if X.shape[0] > 0]
    if not all_data_list:
        print("No data to plot.")
        return
        
    all_data = np.vstack(all_data_list)
    
    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    base_keys = [k for k in data_dict.keys() if "Matched" not in k]
    matched_keys = [k for k in data_dict.keys() if "Matched" in k]
    
    # Colors/Markers for standard groups
    style_map = {
        "RCT Treatment": {"c": "blue", "marker": "o", "alpha": 0.4},
        "RCT Control":   {"c": "green", "marker": "o", "alpha": 0.4},
        "External":      {"c": "orange", "marker": "s", "alpha": 0.3},
        "Matched":       {"c": "black", "marker": "x", "alpha": 0.8}
    }
    
    def get_style(label):
        for key, style in style_map.items():
            if key in label:
                return style
        return {"alpha": 0.5} # Default

    # Plot base layers
    for label in base_keys:
        X = data_dict[label]
        if X.shape[0] == 0:
            continue
        X_pca = pca.transform(X)
        style = get_style(label)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], label=label, **style)

    # Plot matched overlay
    for label in matched_keys:
        X = data_dict[label]
        if X.shape[0] == 0:
            continue
        X_pca = pca.transform(X)
        style = get_style(label)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], label=label, **style)

    plt.title("Covariate Scatter (PC1 vs PC2)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/covariate_scatter_pca.png")
    plt.close()
    print(f"PCA scatter plot saved to {output_dir}/")

def plot_simulation_boxplots(results_list, output_dir="plots"):
    """
    Plots boxplots for Energy and ATT evolution over different N_SAMPLED values.
    results_list: List of dicts, each containing:
        {'n_sample': int, 'energies': [...], 'atts': [...], 'true_tau': float}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_samples = [r['n_sample'] for r in results_list]
    energies_data = [r['energies'] for r in results_list]
    atts_data = [r['atts'] for r in results_list]
    true_tau = results_list[0]['true_tau']

    # 1. Energy Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(energies_data, labels=n_samples)
    plt.title("Evolution of Energy Distance (Bias Proxy)")
    plt.xlabel("N_SAMPLED (Augmentation Size)")
    plt.ylabel("Energy Distance")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/sim_boxplot_energy.png")
    plt.close()

    # 2. ATT Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(atts_data, labels=n_samples)
    plt.axhline(y=true_tau, color='r', linestyle='--', label=f"True ATT ({true_tau})")
    plt.title("Evolution of ATT Estimates")
    plt.xlabel("N_SAMPLED (Augmentation Size)")
    plt.ylabel("Estimated ATT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/sim_boxplot_att.png")
    plt.close()
    
    print(f"Simulation boxplots saved to {output_dir}/")

def plot_mse_evolution(results_list, output_dir="plots"):
    """
    Plots the MSE of ATT estimates as a function of n_sampled.
    results_list: List of dicts, each containing:
        {'n_sample': int, 'energies': [...], 'atts': [...], 'true_tau': float}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_samples = [r['n_sample'] for r in results_list]
    mses = []

    for r in results_list:
        atts = np.array(r['atts'])
        true_tau = r['true_tau']
        # MSE = E[(estimator - true_value)^2]
        mse = np.mean((atts - true_tau) ** 2)
        mses.append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(n_samples, mses, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title("MSE of ATT Estimate vs. Augmentation Size")
    plt.xlabel("N_SAMPLED (Augmentation Size)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/sim_mse_evolution.png")
    plt.close()
    
    print(f"MSE evolution plot saved to {output_dir}/")