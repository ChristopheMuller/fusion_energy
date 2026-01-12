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

def plot_bias_variance_comparison(results_dict, output_dir="plots"):
    """
    Plots Bias and Variance evolution for multiple strategies.
    
    results_dict: { "Method Name": [list of result dicts] }
    Each result dict has: 'n_sample', 'bias', 'variance', etc.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define styles
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'x']
    styles = list(zip(colors, markers))
    
    # Plot Bias
    plt.figure(figsize=(12, 7))
    for i, (method_name, results_list) in enumerate(results_dict.items()):
        n_samples = [r['n_sample'] for r in results_list]
        bias = [np.abs(r['bias']) for r in results_list]
        c, m = styles[i % len(styles)]
        plt.plot(n_samples, bias, marker=m, label=f'{method_name} (|Bias|)', color=c, alpha=0.8)

    plt.title("Absolute Bias Comparison")
    plt.xlabel("N_SAMPLED")
    plt.ylabel("|Bias|")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/bias_comparison.png")
    plt.close()
    
    # Plot Variance
    plt.figure(figsize=(12, 7))
    for i, (method_name, results_list) in enumerate(results_dict.items()):
        n_samples = [r['n_sample'] for r in results_list]
        var = [r['variance'] for r in results_list]
        c, m = styles[i % len(styles)]
        plt.plot(n_samples, var, marker=m, label=f'{method_name} (Variance)', color=c, linestyle='--', alpha=0.8)

    plt.title("Variance Comparison")
    plt.xlabel("N_SAMPLED")
    plt.ylabel("Variance of ATT Estimate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/variance_comparison.png")
    plt.close()
    
    print(f"Bias and Variance comparison plots saved to {output_dir}/")

def plot_energy_comparison(results_dict, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'x']
    styles = list(zip(colors, markers))
    
    plt.figure(figsize=(12, 7))
    
    for i, (method_name, results_list) in enumerate(results_dict.items()):
        n_samples = [r['n_sample'] for r in results_list]
        en = [r['mean_energy'] for r in results_list]
        std = [r['std_energy'] for r in results_list]
        
        c, m = styles[i % len(styles)]
        
        plt.errorbar(n_samples, en, yerr=std, fmt=f'-{m}', color=c, 
                     label=f'{method_name}', capsize=5, alpha=0.8)
    
    plt.title("Energy Distance Comparison (with Std Dev)")
    plt.xlabel("N_SAMPLED")
    plt.ylabel("Energy Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/energy_comparison.png")
    plt.close()
    
    print(f"Energy comparison plot saved to {output_dir}/")

def plot_mse_decomposition_comparison(results_dict, output_dir="plots"):
    """
    Plots MSE decomposition (MSE, Bias^2, Variance) for all methods side-by-side.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6), sharey=True)
    
    if n_methods == 1:
        axes = [axes]
        
    styles = {
        'MSE': {'color': 'purple', 'marker': 'o', 'linestyle': '-', 'linewidth': 2},
        'Bias^2': {'color': 'red', 'marker': 'x', 'linestyle': '--', 'linewidth': 1.5},
        'Variance': {'color': 'green', 'marker': 's', 'linestyle': '--', 'linewidth': 1.5}
    }
    
    for ax, (method_name, results_list) in zip(axes, results_dict.items()):
        n_samples = [r['n_sample'] for r in results_list]
        bias_sq = [r['bias']**2 for r in results_list]
        var = [r['variance'] for r in results_list]
        mse = [b + v for b, v in zip(bias_sq, var)]
        
        ax.plot(n_samples, mse, label='MSE', **styles['MSE'])
        ax.plot(n_samples, bias_sq, label='Bias²', **styles['Bias^2'])
        ax.plot(n_samples, var, label='Variance', **styles['Variance'])
        ax.set_title(f"{method_name}: MSE Decomposition")
        ax.set_xlabel("N_SAMPLED")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    axes[0].set_ylabel("Error Squared")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mse_decomposition_comparison.png")
    plt.close()
    
    print(f"MSE decomposition plot saved to {output_dir}/")

def plot_error_boxplot(raw_errors_dict, output_dir="plots"):
    """
    Plots boxplots of the Estimation Error (Est - True) distribution as n increases for all methods.
    
    raw_errors_dict: { "Method Name": { n_sample: [error_values...] } }
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_methods = len(raw_errors_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6), sharey=True)
    
    if n_methods == 1:
        axes = [axes]
        
    for ax, (method_name, results_map) in zip(axes, raw_errors_dict.items()):
        n_samples = sorted(results_map.keys())
        data = [results_map[n] for n in n_samples]
        
        ax.boxplot(data, labels=n_samples, showfliers=True)
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_title(f"{method_name}: Error Dist")
        ax.set_xlabel("N_SAMPLED")
        if ax == axes[0]:
            ax.set_ylabel("Estimation Error (Est - True)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_boxplot_comparison.png")
    plt.close()
    
    print(f"Error boxplot comparison saved to {output_dir}/")

def plot_energy_mse_method_decomposition(results_dict, output_dir="plots"):
    """
    Plots MSE, Bias^2, Variance (left axis) and Energy (right axis) for each method individually.
    Saves as energy_MSE_***.png
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for method_name, results_list in results_dict.items():
        n_samples = [r['n_sample'] for r in results_list]
        bias_sq = [r['bias']**2 for r in results_list]
        var = [r['variance'] for r in results_list]
        mse = [b + v for b, v in zip(bias_sq, var)]
        energy = [r['mean_energy'] for r in results_list]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Left Axis: Errors
        ax1.set_xlabel("N_SAMPLED")
        ax1.set_ylabel("Error Squared (MSE components)", color='black')
        
        l1 = ax1.plot(n_samples, mse, label='MSE', color='purple', marker='o', linestyle='-', linewidth=2)
        l2 = ax1.plot(n_samples, bias_sq, label='Bias²', color='red', marker='x', linestyle='--', linewidth=0.3)
        l3 = ax1.plot(n_samples, var, label='Variance', color='green', marker='s', linestyle='--', linewidth=0.3)
        
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Right Axis: Energy
        ax2 = ax1.twinx()
        ax2.set_ylabel("Energy Distance", color='blue')
        l4 = ax2.plot(n_samples, energy, label='Energy', color='blue', marker='d', linestyle='-.', linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Combined Legend
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
        
        plt.title(f"Energy & MSE Decomposition: {method_name}")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Sanitize filename
        safe_name = method_name.replace(' ', '_').replace('-', '_')
        filename = f"energy_MSE_{safe_name}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    print(f"Individual Energy & MSE plots saved to {output_dir}/energy_MSE_*.png")
