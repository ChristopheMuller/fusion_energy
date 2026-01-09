import matplotlib.pyplot as plt
import numpy as np
import os

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
    Scatter plot of the first 2 dimensions of covariates.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    
    # Define a rough style map or heuristics
    # We want base layers first, then overlays
    
    # Separating keys to ensure order: Base groups first, then Matches
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
        # Only plot if we have at least 2 dims
        if X.shape[1] < 2:
            continue
            
        style = get_style(label)
        plt.scatter(X[:, 0], X[:, 1], label=label, **style)

    # Plot matched overlay
    for label in matched_keys:
        X = data_dict[label]
        if X.shape[1] < 2:
            continue
            
        style = get_style(label)
        # Make matched points slightly larger or distinct
        plt.scatter(X[:, 0], X[:, 1], label=label, **style)

    plt.title("Covariate Scatter (Dim 0 vs Dim 1)")
    plt.xlabel("Covariate 0")
    plt.ylabel("Covariate 1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/covariate_scatter_2d.png")
    plt.close()
    print(f"Scatter plot saved to {output_dir}/")
