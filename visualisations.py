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

def plot_pca_weights(split_data, est_result, title, filename):
    """
    Computes PCA on RCT data, projects External data, and plots weights.
    
    Args:
        split_data: SplitData object containing X_treat, X_control_int, X_external.
        est_result: EstimationResult object containing weights_external.
        title: Title of the plot.
        filename: Path to save the plot.
    """
    import numpy as np
    
    # Combine RCT data for PCA fitting
    X_treat = split_data.X_treat
    X_control = split_data.X_control_int
    X_rct = np.vstack([X_treat, X_control])
    X_ext = split_data.X_external
    weights = est_result.weights_external
    
    # Compute PCA (using SVD) on the RCT data.
    # To ensure the PCA basis is identical across methods (which only differ in how they split RCT),
    # we sort the RCT data deterministically before fitting PCA.
    # This prevents arbitrary sign flips or numerical differences due to row ordering.
    
    # Lexicographical sort of rows
    sort_idx = np.lexsort(X_rct.T[::-1]) # Sort by last column, then 2nd last...
    X_rct_sorted = X_rct[sort_idx]
    
    # Center the sorted data
    mean_rct = np.mean(X_rct_sorted, axis=0)
    X_rct_centered = X_rct_sorted - mean_rct
    
    # SVD: X = U S Vt. Components are rows of Vt.
    u, s, vt = np.linalg.svd(X_rct_centered, full_matrices=False)
    
    if vt.shape[0] < 2:
        components = np.eye(vt.shape[1])[:, :2]
        if vt.shape[1] == 1:
             components = np.array([[1, 0]]) # 1D to 2D
    else:
        components = vt[:2].T # (n_features, 2)
        
    # Project original (unsorted) data onto the stable components
    # Note: We must use the same mean_rct (which is just the global mean, order invariant)
    proj_treat = (X_treat - mean_rct) @ components
    proj_control = (X_control - mean_rct) @ components
    proj_ext = (X_ext - mean_rct) @ components
    
    plt.figure(figsize=(10, 8))
    
    # Plot RCT Treat
    plt.scatter(proj_treat[:, 0], proj_treat[:, 1], 
                alpha=0.6, color='black', marker='^', label='RCT Treat', s=30)
    
    # Plot RCT Control
    plt.scatter(proj_control[:, 0], proj_control[:, 1], 
                alpha=0.4, color='gray', marker='v', label='RCT Control', s=30)
    
    # Plot External
    # Check if weights are effectively binary (Selection) vs Continuous (Weighting)
    is_binary = np.all(np.isclose(weights, 0) | np.isclose(weights, 1))
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
                        alpha=0.6, color='blue', marker='o', label='External (Selected)', s=40)
            
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
        
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved PCA plot to {filename}")
