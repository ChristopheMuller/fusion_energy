#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import compute_energy_distance_numpy

def run_selection_optimization(n_samples_to_add=20):
    np.random.seed(42)
    
    n_source = 100
    n_target = 100
    n_external = 50
    
    source = np.random.normal(loc=-1.0, scale=1.0, size=(n_source, 1))
    target = np.random.normal(loc=-1.0, scale=1.0, size=(n_target, 1))
    external = np.random.normal(loc=0.0, scale=2.0, size=(n_external, 1))
    
    current_combined = source.copy()
    available_indices = list(range(n_external))
    selected_external_indices = []
    
    initial_distance, _, _, _ = compute_energy_distance_numpy(source, target)
    distance_history = [initial_distance]
    
    for _ in range(n_samples_to_add):
        if not available_indices:
            break
            
        best_candidate_idx = -1
        best_step_distance = float('inf')
        
        for idx in available_indices:
            candidate = external[idx].reshape(1, -1)
            temp_combined = np.vstack([current_combined, candidate])
            
            dist, _, _, _ = compute_energy_distance_numpy(temp_combined, target)
            
            if dist < best_step_distance:
                best_step_distance = dist
                best_candidate_idx = idx
        
        if best_candidate_idx != -1:
            selected_external_indices.append(best_candidate_idx)
            available_indices.remove(best_candidate_idx)
            current_combined = np.vstack([current_combined, external[best_candidate_idx].reshape(1, -1)])
            distance_history.append(best_step_distance)
            
    return source, target, external, current_combined, distance_history

def plot_results(source, target, external, final_combined, history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot(range(len(history)), history, marker='o', linestyle='-')
    axes[0].set_title('Energy Distance vs Iterations')
    axes[0].set_xlabel('Number of External Units Added')
    axes[0].set_ylabel('Energy Distance')
    axes[0].grid(True, alpha=0.3)
    
    sns.kdeplot(source.ravel(), ax=axes[1], label='Source (Initial)', fill=True, alpha=0.3)
    sns.kdeplot(target.ravel(), ax=axes[1], label='Target', fill=True, alpha=0.3)
    sns.kdeplot(external.ravel(), ax=axes[1], label='External Pool', linestyle='--')
    sns.kdeplot(final_combined.ravel(), ax=axes[1], label='Source + Selected (Final)', color='red', linewidth=2)
    
    axes[1].set_title('Distribution Evolution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    src, tgt, ext, final, dist_hist = run_selection_optimization(n_samples_to_add=25)
    print(f"Optimization complete. Final Energy Distance: {dist_hist[-1]}")
    plot_results(src, tgt, ext, final, dist_hist)
#%%