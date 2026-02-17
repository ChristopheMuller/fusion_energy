import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, Any

def plot_error_boxplots(logs: Dict[str, Any], filename="plots/error_boxplots.png"):
    """
    Generates a boxplot of errors for each method.
    
    Args:
        logs: Dictionary mapping method names to SimLog objects (or objects with a .results list of EstimationResult).
        filename: Path to save the plot.
    """
    data = []
    for method_name, log in logs.items():
        for res in log.results:
            data.append({
                "Method": method_name,
                "Error": res.bias 
            })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="Method", y="Error")
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.title("Error Distribution across Simulations")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved boxplot to {filename}")

def plot_mse_decomposition(logs: Dict[str, Any], filename="plots/mse_decomposition.png"):
    """
    Generates a stacked bar chart of MSE decomposed into Squared Bias and Variance.
    
    Args:
        logs: Dictionary mapping method names to SimLog objects.
        filename: Path to save the plot.
    """
    import numpy as np
    
    data = []
    for method_name, log in logs.items():
        # Extract results
        errors = np.array([res.bias for res in log.results])
        estimates = np.array([res.ate_est for res in log.results])
        
        # Compute metrics
        bias_val = np.mean(errors)
        squared_bias = bias_val**2
        variance = np.var(estimates)
        mse = squared_bias + variance
        
        data.append({
            "Method": method_name,
            "Squared Bias": squared_bias,
            "Variance": variance,
            "MSE": mse
        })
    
    df = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Create stacked bars
    # We plot Variance at the bottom, Squared Bias on top
    p1 = plt.bar(df["Method"], df["Variance"], label='Variance', color='skyblue', alpha=0.8)
    p2 = plt.bar(df["Method"], df["Squared Bias"], bottom=df["Variance"], label='Squared Bias', color='salmon', alpha=0.8)
    
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Decomposition: Squared Bias + Variance')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add text labels for MSE
    for i, row in enumerate(df.itertuples()):
        total_height = row.MSE
        plt.text(i, total_height + (0.01 * max(df["MSE"])), f"{total_height:.3f}", 
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved MSE decomposition plot to {filename}")

def plot_pca_weights(split_data, est_result, title, filename):
    """
    Computes PCA on RCT data, projects External data, and plots weights.
    Runs twice: once for internal weights and once for external weights.
    
    Args:
        split_data: SplitData object containing X_treat, X_control_int, X_external.
        est_result: EstimationResult object containing weights_external and weights_continuous.
        title: Title of the plot.
        filename: Path to save the plot.
    """
    import numpy as np
    
    # Combine RCT data for PCA fitting
    X_treat = split_data.X_treat
    X_control = split_data.X_control_int
    X_rct = np.vstack([X_treat, X_control])
    X_ext = split_data.X_external
    
    # Compute PCA (using SVD) on the RCT data.
    # To ensure the PCA basis is identical across methods (which only differ in how they split RCT),
    # we sort the RCT data deterministically before fitting PCA.
    
    # Lexicographical sort of rows
    sort_idx = np.lexsort(X_rct.T[::-1]) # Sort by last column, then 2nd last...
    X_rct_sorted = X_rct[sort_idx]
    
    # Center the sorted data
    mean_rct = np.mean(X_rct_sorted, axis=0)
    X_rct_centered = X_rct_sorted - mean_rct
    
    # SVD: X = U S Vt. Components are rows of Vt.
    u, s, vt = np.linalg.svd(X_rct_centered, full_matrices=False)
    
    # Calculate explained variance ratio
    explained_variance = (s**2) / np.sum(s**2)
    var_pc1 = explained_variance[0] * 100 if len(explained_variance) > 0 else 0
    var_pc2 = explained_variance[1] * 100 if len(explained_variance) > 1 else 0
    
    if vt.shape[0] < 2:
        components = np.eye(vt.shape[1])[:, :2]
        if vt.shape[1] == 1:
             components = np.array([[1, 0]]) # 1D to 2D
    else:
        components = vt[:2].T # (n_features, 2)
        
    # Project original (unsorted) data onto the stable components
    proj_treat = (X_treat - mean_rct) @ components
    proj_control = (X_control - mean_rct) @ components
    proj_ext = (X_ext - mean_rct) @ components
    
    base_filename, ext = os.path.splitext(filename)
    
    weight_sets = {
        "external": est_result.weights_external,
        "continuous": est_result.weights_continuous
    }
    
    for suffix, weights in weight_sets.items():
        plt.figure(figsize=(10, 8))
        
        # Plot RCT Treat
        plt.scatter(proj_treat[:, 0], proj_treat[:, 1], 
                    alpha=0.6, color='black', marker='^', label='RCT Treat', s=30)
        
        # Plot RCT Control
        plt.scatter(proj_control[:, 0], proj_control[:, 1], 
                    alpha=0.4, color='gray', marker='v', label='RCT Control', s=30)
        
        # Plot External
        unique_vals = np.unique(np.round(weights, 5))
        
        # Heuristic: if many unique weights -> continuous. If few -> matching.
        if len(unique_vals) <= 5: 
            # Treated as selection/matching
            # Selected (w > 0)
            selected_mask = weights > 1e-6
            not_selected_mask = ~selected_mask
            
            # Plot not selected
            if np.any(not_selected_mask):
                plt.scatter(proj_ext[not_selected_mask, 0], proj_ext[not_selected_mask, 1], 
                            alpha=0.2, color='red', marker='x', label='External (Ignored)', s=20)
            
            # Plot selected
            if np.any(selected_mask):
                plt.scatter(proj_ext[selected_mask, 0], proj_ext[selected_mask, 1], 
                            alpha=0.6, color='blue', marker='o', label=f'External (Selected, n={np.sum(selected_mask)})', s=40)
                
        else:
            # Continuous weighting
            # Normalize weights for size:
            if np.sum(weights) > 0:
                norm_w = weights / np.mean(weights)
                sizes = norm_w * 20
            else:
                sizes = 20
                
            plt.scatter(proj_ext[:, 0], proj_ext[:, 1], 
                        alpha=0.4, color='blue', label='External (Weighted)', s=sizes)
            
        plt.title(f"{title} ({suffix.capitalize()})")
        plt.xlabel(f"PC1 ({var_pc1:.1f}%)")
        plt.ylabel(f"PC2 ({var_pc2:.1f}%)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        out_path = f"{base_filename}_{suffix}{ext}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved PCA plot to {out_path}")


def plot_energy_distance(logs: Dict[str, Any], filename="plots/energy_distance.png"):
    """
    Generates a bar plot of average Energy Distance for each method.

    Args:
        logs: Dictionary mapping method names to SimLog objects.
        filename: Path to save the plot.
    """
    import numpy as np

    data = []
    for method_name, log in logs.items():
        # Extract results
        energies = np.array([res.energy_distance for res in log.results])
        
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        data.append({
            "Method": method_name,
            "Mean Energy Distance": mean_energy,
            "Std Energy Distance": std_energy
        })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    
    # Bar plot with error bars
    plt.bar(df["Method"], df["Mean Energy Distance"], yerr=df["Std Energy Distance"], 
            capsize=5, color='mediumpurple', alpha=0.8)
    
    plt.ylabel('Mean Energy Distance')
    plt.title('Average Energy Distance (Target vs. Pooled Control)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add text labels
    for i, height in enumerate(df["Mean Energy Distance"]):
        plt.text(i, height + (0.01 * max(df["Mean Energy Distance"])), f"{height:.4f}", 
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved Energy Distance plot to {filename}")


def plot_metric_curves(logs: Dict[str, Any], filename="plots/metric_curves.png"):
    """
    Generates a line plot of Bias^2, Variance, MSE, and Energy Distance
    against the average number of external data points used.

    Args:
        logs: Dictionary mapping method names to SimLog objects.
        filename: Path to save the plot.
    """
    import numpy as np
    
    data = []
    for method_name, log in logs.items():
        # Extract results
        errors = np.array([res.bias for res in log.results])
        estimates = np.array([res.ate_est for res in log.results])
        n_exts = np.array([res.n_external_used() for res in log.results])
        energies = np.array([res.energy_distance for res in log.results])
        # energy_progs = np.array([res.energy_distance_prognostic for res in log.results if res.energy_distance_prognostic is not None])
        
        # Compute metrics
        avg_n_ext = np.mean(n_exts)
        bias_val = np.mean(errors)
        squared_bias = bias_val**2
        variance = np.var(estimates)
        mse = squared_bias + variance
        avg_energy = np.mean(energies)
        
        data.append({
            "Method": method_name,
            "Avg_N_Ext": avg_n_ext,
            "Squared Bias": squared_bias,
            "Variance": variance,
            "MSE": mse,
            "Energy": avg_energy,
            # "Energy_Prognostic": np.mean(energy_progs) if energy_progs.size > 0 else 0.0
        })
    
    df = pd.DataFrame(data).sort_values("Avg_N_Ext")
    
    # Identify minima
    min_mse_row = df.loc[df["MSE"].idxmin()]
    min_energy_row = df.loc[df["Energy"].idxmin()]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot MSE components on left axis
    ax1.plot(df["Avg_N_Ext"], df["Squared Bias"], marker='o', label='Bias^2', color='salmon')
    ax1.plot(df["Avg_N_Ext"], df["Variance"], marker='s', label='Variance', color='skyblue')
    ax1.plot(df["Avg_N_Ext"], df["MSE"], marker='d', label='MSE', color='green', linewidth=2)
    
    # Highlight Min MSE
    ax1.scatter(min_mse_row["Avg_N_Ext"], min_mse_row["MSE"], 
                color='darkgreen', s=200, zorder=10, marker='*', 
                label=f'Min MSE ({min_mse_row["Avg_N_Ext"]:.1f})')

    ax1.set_xlabel('Average External Data Points (Sum of weights)')
    ax1.set_ylabel('MSE Metrics')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Create twin axis for Energy Distance
    ax2 = ax1.twinx()
    ax2.plot(df["Avg_N_Ext"], df["Energy"], marker='x', label='Energy Distance', color='mediumpurple', linestyle='--')
    
    # Highlight Min Energy
    ax2.scatter(min_energy_row["Avg_N_Ext"], min_energy_row["Energy"], 
                color='indigo', s=200, zorder=10, marker='*', 
                label=f'Min Energy ({min_energy_row["Avg_N_Ext"]:.1f})')

    ax2.set_ylabel('Energy Distance', color='mediumpurple')
    ax2.tick_params(axis='y', labelcolor='mediumpurple')

    # if df["Energy_Prognostic"].any():
    #     ax2.plot(df["Avg_N_Ext"], df["Energy_Prognostic"]/10, marker='x', label='Energy Distance (Prognostic)', color='orange', linestyle='--')
    
    plt.title('Performance Metrics vs. External Data Sample Size')
    
    # Unified legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved metric curves plot to {filename}")

def plot_weight_ranks(est_result, title, filename, n_external=None):
    """
    Plots the continuous weights ranked from largest to smallest.
    
    Args:
        est_result: EstimationResult object containing weights_continuous.
        title: Title of the plot.
        filename: Path to save the plot.
    """
    import numpy as np
    
    weights = est_result.weights_continuous
    
    # Sort weights descending
    sorted_weights = np.sort(weights)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_weights)), sorted_weights, marker='o', linestyle='-', markersize=2, alpha=0.7)

    if n_external is not None:
        plt.axvline(n_external - 1, color='red', linestyle='--', label=f'Target n_external = {n_external}')
    
    plt.title(f"{title} - Weight Ranks")
    plt.xlabel("Rank")
    plt.ylabel("Weight Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved weight rank plot to {filename}")

def plot_estimation_time(logs: Dict[str, Any], filename="plots/estimation_time.png"):
    """
    Generates a bar plot of average estimation time with confidence intervals for each method.
    
    Args:
        logs: Dictionary mapping method names to SimLog objects.
        filename: Path to save the plot.
    """
    data = []
    for method_name, log in logs.items():
        for res in log.results:
            data.append({
                "Method": method_name,
                "Estimation Time (s)": res.estimation_time
            })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x="Method", y="Estimation Time (s)", capsize=.2)
    
    plt.title("Average Estimation Time per Method")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved estimation time plot to {filename}")
