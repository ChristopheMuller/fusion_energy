import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

def plot_covariate_and_outcome_distributions(data, save_path=None):
    X_treat = data["rct_treat"]["X"]
    X_control = data["rct_control"]["X"]
    X_ext = data["external"]["X"]
    
    Y_treat = data["rct_treat"]["Y"]
    Y_control = data["rct_control"]["Y"]
    Y_ext = data["external"]["Y"]
    
    dim = X_treat.shape[1]
    
    if dim > 2:
        pca = PCA(n_components=2)
        X_all = np.vstack([X_treat, X_control, X_ext])
        pca.fit(X_all)
        
        X_treat_2d = pca.transform(X_treat)
        X_control_2d = pca.transform(X_control)
        X_ext_2d = pca.transform(X_ext)
        
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)"
    elif dim == 2:
        X_treat_2d = X_treat
        X_control_2d = X_control
        X_ext_2d = X_ext
        xlabel = "Covariate 1"
        ylabel = "Covariate 2"
    else:
        X_treat_2d = np.column_stack([X_treat, np.zeros_like(X_treat)])
        X_control_2d = np.column_stack([X_control, np.zeros_like(X_control)])
        X_ext_2d = np.column_stack([X_ext, np.zeros_like(X_ext)])
        xlabel = "Covariate 1"
        ylabel = ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(X_ext_2d[:, 0], X_ext_2d[:, 1], 
                alpha=0.3, label='External Control', color='gray', s=20)
    ax1.scatter(X_control_2d[:, 0], X_control_2d[:, 1], 
                alpha=0.6, label='RCT Control', color='blue', s=30)
    ax1.scatter(X_treat_2d[:, 0], X_treat_2d[:, 1], 
                alpha=0.6, label='RCT Treat', color='red', s=30)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title("Covariate Distribution (PCA Projection)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    def plot_kde(ax, data, color, label):
        density = gaussian_kde(data)
        xs = np.linspace(min(data), max(data), 200)
        ax.plot(xs, density(xs), color=color, label=label)
        ax.fill_between(xs, density(xs), color=color, alpha=0.2)

    plot_kde(ax2, Y_ext, 'gray', 'External Control')
    plot_kde(ax2, Y_control, 'blue', 'RCT Control')
    plot_kde(ax2, Y_treat, 'red', 'RCT Treat')
    
    ax2.set_xlabel("Outcome (Y)")
    ax2.set_ylabel("Density")
    ax2.set_title("Outcome Distributions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_post_matching_covariates(data, selected_indices, save_path=None):
    X_treat = data["rct_treat"]["X"]
    X_control = data["rct_control"]["X"]
    X_ext = data["external"]["X"]
    
    n_control = X_control.shape[0]
    dim = X_treat.shape[1]
    
    # PCA or Projection
    if dim > 2:
        pca = PCA(n_components=2)
        X_all = np.vstack([X_treat, X_control, X_ext])
        pca.fit(X_all)
        
        X_treat_2d = pca.transform(X_treat)
        X_control_2d = pca.transform(X_control)
        X_ext_2d = pca.transform(X_ext)
        
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)"
    elif dim == 2:
        X_treat_2d = X_treat
        X_control_2d = X_control
        X_ext_2d = X_ext
        xlabel = "Covariate 1"
        ylabel = "Covariate 2"
    else:
        X_treat_2d = np.column_stack([X_treat, np.zeros_like(X_treat)])
        X_control_2d = np.column_stack([X_control, np.zeros_like(X_control)])
        X_ext_2d = np.column_stack([X_ext, np.zeros_like(X_ext)])
        xlabel = "Covariate 1"
        ylabel = ""

    plt.figure(figsize=(10, 8))
    
    # Background: Original Points (Faint)
    plt.scatter(X_ext_2d[:, 0], X_ext_2d[:, 1], 
                alpha=0.1, label='External (Unselected)', color='gray', s=20)
    plt.scatter(X_control_2d[:, 0], X_control_2d[:, 1], 
                alpha=0.2, label='RCT Control (Unselected)', color='blue', s=20)
    
    # Process Selections
    unique_indices, counts = np.unique(selected_indices, return_counts=True)
    
    # Base size multiplier
    size_factor = 50
    sizes = counts * size_factor
    
    # Split indices
    mask_control = unique_indices < n_control
    mask_ext = unique_indices >= n_control
    
    # Selected RCT Control
    idx_sel_control = unique_indices[mask_control]
    sizes_control = sizes[mask_control]
    if len(idx_sel_control) > 0:
        plt.scatter(X_control_2d[idx_sel_control, 0], X_control_2d[idx_sel_control, 1],
                    s=sizes_control, alpha=0.7, label='RCT Control (Selected)', 
                    color='blue', edgecolors='black', linewidth=1)
        
    # Selected External -> GREEN
    idx_sel_ext = unique_indices[mask_ext] - n_control
    sizes_ext = sizes[mask_ext]
    if len(idx_sel_ext) > 0:
        plt.scatter(X_ext_2d[idx_sel_ext, 0], X_ext_2d[idx_sel_ext, 1],
                    s=sizes_ext, alpha=0.8, label='External (Selected)', 
                    color='green', edgecolors='black', linewidth=1)

    # RCT Treat (Target) - Always prominent
    plt.scatter(X_treat_2d[:, 0], X_treat_2d[:, 1], 
                alpha=0.6, label='RCT Treat', color='red', s=40, marker='D')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Covariate Distribution After Matching (Weighted Selection)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_matched_outcomes(data, Y_sample, save_path=None):

    Y_treat = data["rct_treat"]["Y"]

    Y_control = data["rct_control"]["Y"]

    Y_ext = data["external"]["Y"]

    

    plt.figure(figsize=(10, 6))

    

    def plot_kde(ax, data, color, label, linestyle='-'):

        density = gaussian_kde(data)

        xs = np.linspace(min(data), max(data), 200)

        ax.plot(xs, density(xs), color=color, label=label, linestyle=linestyle, linewidth=2)

        ax.fill_between(xs, density(xs), color=color, alpha=0.1)



    ax = plt.gca()

    plot_kde(ax, Y_ext, 'gray', 'External Control')

    plot_kde(ax, Y_control, 'blue', 'RCT Control')

    plot_kde(ax, Y_treat, 'red', 'RCT Treat')

    plot_kde(ax, Y_sample, 'green', 'Matched Sample', linestyle='--')

    

    plt.xlabel("Outcome (Y)")

    plt.ylabel("Density")

    plt.title("Outcome Distributions After Matching")

    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_weight_distribution(weights, save_path=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(weights, bins=50, color='purple', alpha=0.7)
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.title("Histogram of Optimal Weights")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.sort(weights)[::-1], color='purple', linewidth=2)
    plt.xlabel("Rank")
    plt.ylabel("Weight Value")
    plt.title("Sorted Weights (Sparsity Check)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()