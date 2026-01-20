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
            if "Matched" in label or "Fused" in label:
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
        if "Matched" in label or "Fused" in label:
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
    all_data_rct = []
    for key in data_dict.keys():
        if "RCT" in key:
            all_data_rct.append(data_dict[key])

    all_data = np.vstack(all_data_rct)

    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    base_keys = [k for k in data_dict.keys() if "Matched" not in k and "Fused" not in k]
    matched_keys = [k for k in data_dict.keys() if "Matched" in k or "Fused" in k]
    
    # Colors/Markers for standard groups
    style_map = {
        "RCT Treatment": {"c": "blue", "marker": "o", "alpha": 0.4},
        "RCT Control":   {"c": "green", "marker": "o", "alpha": 0.4},
        "External":      {"c": "orange", "marker": "s", "alpha": 0.3},
        "Matched":       {"c": "black", "marker": "x", "alpha": 0.8},
        "Fused":         {"c": "black", "marker": "x", "alpha": 0.8}
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
    n_matched = 0
    for label in matched_keys:
        X = data_dict[label]
        if X.shape[0] == 0:
            continue
        X_pca = pca.transform(X)
        style = get_style(label)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], label=label, **style)

        if "Matched" in label or "Fused" in label:
            n_matched = X.shape[0]


    plt.title(f"Covariate Scatter (PC1 vs PC2, n={n_matched})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/covariate_scatter_pca.png")
    plt.close()
    print(f"PCA scatter plot saved to {output_dir}/")

def plot_covariate_2d_scatter_weighted(data_dict, weights, output_dir="plots"):
    """
    Scatter plot of the first 2 Principal Components (PCA).
    Points are sized according to 'weights'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    
    # Concatenate all data to fit PCA on the global distribution space
    all_data_rct = []
    for key in data_dict.keys():
        if "RCT" in key:
            all_data_rct.append(data_dict[key])

    all_data = np.vstack(all_data_rct)
    
    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    base_keys = [k for k in data_dict.keys() if "Matched" not in k and "Fused" not in k]
    matched_keys = [k for k in data_dict.keys() if "Matched" in k or "Fused" in k]
    
    # Colors/Markers for standard groups
    style_map = {
        "RCT Treatment": {"c": "blue", "marker": "o", "alpha": 0.4},
        "RCT Control":   {"c": "green", "marker": "o", "alpha": 0.4},
        "External":      {"c": "orange", "marker": "s", "alpha": 0.3},
        "Matched":       {"c": "black", "marker": "x", "alpha": 0.8},
        "Fused":         {"c": "black", "marker": "x", "alpha": 0.8}
    }
    
    def get_style(label):
        for key, style in style_map.items():
            if key in label:
                return style
        return {"alpha": 0.5} # Default

    # Helper to get subset weights
    X_i = data_dict.get("RCT Control")
    X_e = data_dict.get("External")
    n_i = X_i.shape[0] if X_i is not None else 0
    n_e = X_e.shape[0] if X_e is not None else 0
    
    # Plot base layers
    for label in base_keys:
        X = data_dict[label]
        if X.shape[0] == 0:
            continue
        X_pca = pca.transform(X)
        style = get_style(label)
        
        # Calculate sizes
        s = 30 # Default size
        
        if weights is not None:
            w_subset = None
            if label == "External":
                if len(weights) == n_e:
                    w_subset = weights
                elif len(weights) == n_i + n_e:
                    w_subset = weights[n_i:]
            elif label == "RCT Control":
                if len(weights) == n_i + n_e:
                    w_subset = weights[:n_i]
            
            if w_subset is not None and len(w_subset) > 0:
                # Normalize for visibility: mean size 30
                # Avoid div by zero
                w_mean = np.mean(w_subset)
                if w_mean > 1e-9:
                    s = 30 * (w_subset / w_mean)
                else:
                    s = 30 # Fallback
                
                # Clip extremely large sizes
                s = np.clip(s, 1, 300)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=s, label=label, **style)

    # Plot matched overlay
    n_matched = 0
    for label in matched_keys:
        X = data_dict[label]
        if X.shape[0] == 0:
            continue
        X_pca = pca.transform(X)
        style = get_style(label)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], label=label, **style)
        if "Matched" in label or "Fused" in label:
            n_matched = X.shape[0]

    plt.title(f"Weighted Covariate Scatter (PC1 vs PC2, n={n_matched})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/covariate_scatter_weighted_pca.png")
    plt.close()
    print(f"Weighted PCA scatter plot saved to {output_dir}/")

def plot_treatment_effect_heterogeneity(X, true_tau_values, output_dir="plots"):
    tau_dir = os.path.join(output_dir, "tau")
    if not os.path.exists(tau_dir):
        os.makedirs(tau_dir)
    
    dim = X.shape[1]
    for d in range(dim):
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, d], true_tau_values, alpha=0.5, c='purple')
        plt.title(f"Heterogeneity of Treatment Effect (vs Covariate {d+1})")
        plt.xlabel(f"Covariate {d+1}")
        plt.ylabel("True Treatment Effect (Tau)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{tau_dir}/tau_heterogeneity_cov_{d+1}.png")
        plt.close()
    
    print(f"Tau heterogeneity plots saved to {tau_dir}/")

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
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    lines = []
    
    for i, (method_name, results_list) in enumerate(results_dict.items()):
        n_samples = [r['n_sample'] for r in results_list]
        en = [r['mean_energy'] for r in results_list]
        std = [r['std_energy'] for r in results_list]
        
        c, m = styles[i % len(styles)]
        
        # Plot Total Energy on Left Axis
        l_en = ax1.errorbar(n_samples, en, yerr=std, fmt=f'-{m}', color=c, 
                     label=f'{method_name} (Total Energy)', capsize=5, alpha=0.8, linewidth=2)
        lines.append(l_en)
        
        # Add decomposition on Right Axis if available
        if all(k in results_list[0] for k in ['mean_d12', 'mean_d11']):
            d12 = [r['mean_d12'] for r in results_list]
            d11 = [r['mean_d11'] for r in results_list]
            
            l_d12, = ax2.plot(n_samples, d12, linestyle='--', color=c, alpha=0.5, label=f'{method_name}: d12')
            l_d11, = ax2.plot(n_samples, d11, linestyle=':', color=c, alpha=0.5, label=f'{method_name}: d11')
            lines.extend([l_d12, l_d11])
    
    ax1.set_xlabel("N_SAMPLED")
    ax1.set_ylabel("Total Energy Distance")
    ax2.set_ylabel("Component Distances (d12, d11)")
    
    plt.title("Energy Distance Comparison & Decomposition")
    
    # Combined Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='small', ncol=2)
    
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/energy_comparison.png")
    plt.close()
    
    print(f"Energy comparison plot saved to {output_dir}/")

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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for method_name, results_list in results_dict.items():
        n_samples = [r['n_sample'] for r in results_list]
        bias_sq = [r['bias']**2 for r in results_list]
        var = [r['variance'] for r in results_list]
        mse = [b + v for b, v in zip(bias_sq, var)]
        energy = [r['mean_energy'] for r in results_list]
        
        # New: Pooled Energy
        pooled_en = [r.get('mean_pooled_energy', np.nan) for r in results_list]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel("N_SAMPLED")
        ax1.set_ylabel("Error Squared (MSE components)", color='black')
        
        l1 = ax1.plot(n_samples, mse, label='MSE', color='purple', marker='o', linestyle='-', linewidth=2)
        l2 = ax1.plot(n_samples, bias_sq, label='Bias²', color='red', marker='x', linestyle='--', linewidth=0.3)
        l3 = ax1.plot(n_samples, var, label='Variance', color='green', marker='s', linestyle='--', linewidth=0.3)
        
        ax1.tick_params(axis='y', labelcolor='black')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Energy Distance", color='blue')
        l4 = ax2.plot(n_samples, energy, label='Energy (Current)', color='blue', marker='d', linestyle='-.', linewidth=1.5)
        
        # Plot Pooled Energy Reference
        l5 = ax2.plot(n_samples, pooled_en, label='Energy (Pooled RCT)', color='black', marker='*', linestyle=':', linewidth=2)
        
        ax2.tick_params(axis='y', labelcolor='blue')
        
        lines = l1 + l2 + l3 + l4 + l5
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        
        plt.title(f"Energy & MSE Decomposition: {method_name}")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"energy_MSE.png"
        out_dir_method = os.path.join(output_dir, method_name)
        if not os.path.exists(out_dir_method):
            os.makedirs(out_dir_method)
        plt.savefig(os.path.join(out_dir_method, filename))
        plt.close()
        
    n_methods = len(results_dict)
    if n_methods > 0:
        methods_sorted = sorted(results_dict.keys())

        width_ratios = []
        for m in methods_sorted:
            res = results_dict[m]
            bsq = np.array([r['bias']**2 for r in res])
            var = np.array([r['variance'] for r in res])
            mse_chk = bsq + var
            
            # If constant (std ~ 0), it is a weighting method (independent of n_sampled)
            if len(mse_chk) > 1 and np.std(mse_chk) < 1e-8:
                width_ratios.append(2.5)
            else:
                width_ratios.append(5.5)
        total_width = sum(width_ratios)
        fig, axes = plt.subplots(1, n_methods, figsize=(total_width, 5), sharey=True,
                                 gridspec_kw={'width_ratios': width_ratios})
        if n_methods == 1:
            axes = [axes]
        
        all_energies = []
        for m in methods_sorted:
            engs = [r['mean_energy'] for r in results_dict[m] if r['mean_energy'] is not None and not np.isnan(r['mean_energy'])]
            pool_engs = [r.get('mean_pooled_energy', np.nan) for r in results_dict[m]]
            all_energies.extend(engs)
            all_energies.extend([e for e in pool_engs if not np.isnan(e)])
            
        if all_energies:
            min_en, max_en = min(all_energies), max(all_energies)
            pad = (max_en - min_en) * 0.1 if max_en != min_en else 1.0
            ylim_en = (min_en - pad, max_en + pad)
        else:
            ylim_en = (0, 1)

        for i, (ax1, method_name) in enumerate(zip(axes, methods_sorted)):
            results_list = results_dict[method_name]
            n_samples = [r['n_sample'] for r in results_list]
            bias_sq = [r['bias']**2 for r in results_list]
            var = [r['variance'] for r in results_list]
            mse = [b + v for b, v in zip(bias_sq, var)]
            energy = [r['mean_energy'] for r in results_list]
            pooled_en = [r.get('mean_pooled_energy', np.nan) for r in results_list]
            
            ax1.set_xlabel("Samples")
            if i == 0:
                ax1.set_ylabel("Error Squared (MSE components)", color='black')
            
            l1 = ax1.plot(n_samples, mse, label='MSE', color='purple', marker='o', linestyle='-', linewidth=2)
            l2 = ax1.plot(n_samples, bias_sq, label='Bias²', color='red', marker='x', linestyle='--', linewidth=0.3)
            l3 = ax1.plot(n_samples, var, label='Variance', color='green', marker='s', linestyle='--', linewidth=0.3)
            
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(method_name)
            
            ax2 = ax1.twinx()
            l4 = ax2.plot(n_samples, energy, label='Energy (Current)', color='blue', marker='d', linestyle='-.', linewidth=1.5)
            l5 = ax2.plot(n_samples, pooled_en, label='Energy (Pooled)', color='black', marker='*', linestyle=':', linewidth=1.5)
            
            ax2.set_ylim(ylim_en)
            ax2.tick_params(axis='y', labelcolor='blue')
            
            if i < n_methods - 1:
                ax2.set_yticklabels([])
            else:
                ax2.set_ylabel("Energy Distance", color='blue')

        lines = l1 + l2 + l3 + l4 + l5
        labels = [l.get_label() for l in lines]
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "energy_MSE_combined.png"), bbox_inches='tight')
        plt.close()
        print(f"Combined Energy & MSE plot saved to {output_dir}/energy_MSE_combined.png")