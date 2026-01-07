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
