import numpy as np
import matplotlib.pyplot as plt
import os
from data_gen import create_complex_dataset
from methods.methods_weighted import EnergyAugmenter_Weighted

N_TREAT = 100
N_RCT_CONTROL = 100
N_EXT = 1000
DIM = 10
TAU = 1.0

EXT_BIAS = 1
EXT_VAR = 1.5
RCT_VAR = 1.0
N_SAMPLED = 100

print(f"--- Weight Investigation ---")
print(f"N_TREAT: {N_TREAT}, N_RCT_CONTROL: {N_RCT_CONTROL}, N_EXT: {N_EXT}")
print(f"DIM: {DIM}, EXT_BIAS: {EXT_BIAS}, EXT_VAR: {EXT_VAR}, RCT_VAR: {RCT_VAR}")
print(f"N_SAMPLED: {N_SAMPLED}")

# 1. Generate Data
data = create_complex_dataset(
    n_treat=N_TREAT,
    n_control_rct=N_RCT_CONTROL,
    n_external=N_EXT,
    dim=DIM,
    tau=TAU,
    rct_bias=0.,
    ext_bias=EXT_BIAS,
    ext_var=EXT_VAR,
    rct_var=RCT_VAR
)

X_target = data["target"]["X"]
X_internal = data["internal"]["X"]
X_external = data["external"]["X"]

# 2. Fit Weighting Model
print("Fitting EnergyAugmenter_Weighted...")
augmenter = EnergyAugmenter_Weighted(n_sampled=N_SAMPLED, n_iter=1000, lr=0.01)
augmenter.fit(X_target, X_internal, X_external)

weights = augmenter.weights_

# --- Weight Summary Statistics ---
max_w = np.max(weights)
min_w = np.min(weights)
ess = 1.0 / np.sum(weights**2)
near_zero = np.sum(weights < 1e-4)

print(f"\nWeight Statistics:")
print(f"Max Weight: {max_w:.6f}")
print(f"Min Weight: {min_w:.6f}")
print(f"Effective Sample Size (ESS): {ess:.2f} (out of {N_EXT})")
print(f"Weights < 1e-4: {near_zero} ({near_zero/N_EXT:.1%} of external samples)")
print(f"Top 5 weights sum: {np.sum(np.sort(weights)[-5:]):.4f}")

# 3. Visualization
if not os.path.exists("plots"):
    os.makedirs("plots")
    
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot 1: Histogram of Weights ---
axes[0].hist(weights, bins=50, color='skyblue', edgecolor='black')
axes[0].set_title("Distribution of Weights (External Samples)")
axes[0].set_xlabel("Weight")
axes[0].set_ylabel("Frequency")
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Weight vs Distance to Target Mean ---
target_mean = X_target.mean(axis=0)
dists = np.linalg.norm(X_external - target_mean, axis=1)

axes[1].scatter(dists, weights, alpha=0.5, c=weights, cmap='viridis')
axes[1].set_title("Weight vs Distance to Target Mean")
axes[1].set_xlabel("Euclidean Distance to Target Mean")
axes[1].set_ylabel("Weight")
axes[1].grid(True, alpha=0.3)

# --- Plot 3: Weights in Covariate Space (First 2 Dims) ---
sc = axes[2].scatter(X_external[:, 0], X_external[:, 1], 
                    c=weights, s=weights * (N_EXT * 20), # Scale size for visibility
                    cmap='viridis', alpha=0.6, label='External Samples')

# Overlay Target Samples
axes[2].scatter(X_target[:, 0], X_target[:, 1], 
                c='red', s=10, alpha=0.2, label='Target (RCT Treat)')

# Overlay Target Mean
axes[2].scatter(target_mean[0], target_mean[1], 
                c='black', marker='X', s=150, label='Target Mean')

plt.colorbar(sc, ax=axes[2], label='Weight')
axes[2].set_title("Weights in Covariate Space")
axes[2].set_xlabel("Dimension 1")
axes[2].set_ylabel("Dimension 2")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_path = "plots/weight_investigation.png"
plt.savefig(output_path)
print(f"Plots saved to {output_path}")
