#%%

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from structures import EstimationResult
from generators import DataGenerator
from design import FixedRatioDesign, EnergyOptimisedDesign
from estimator import EnergyMatchingEstimator, DummyMatchingEstimator, EnergyWeightingEstimator, OptimalEnergyMatchingEstimator, IPWEstimator, \
    EnergyPooledWeightingEstimator, PrognosticEnergyWeightingEstimator, PrognosticEnergyMatchingEstimator
from dataclasses import dataclass, field
from typing import List, Any

from metrics import compute_weighted_energy
from visualisations import (
    plot_error_boxplots, 
    plot_pca_weights, 
    plot_mse_decomposition, 
    plot_energy_distance,
    plot_metric_curves,
    plot_weight_ranks,
    plot_estimation_time
)

from torch import cdist
import torch

#%%

dim = 5
n_rct = 100
n_ext = 5000
mean_rct = np.zeros(dim)
var_rct = 1
bias_ext = 0.7
var_ext = 1.5
corr = 0.4
beta_bias_ext = 0

treatment_effect = 2



beta = np.ones(dim)

gen = DataGenerator(dim=dim, beta=beta)

# 1. Generate Data
rct_data = gen.generate_rct_pool(n=n_rct, mean=mean_rct, var=var_rct, corr=corr, treatment_effect=treatment_effect)
ext_data = gen.generate_external_pool(n=n_ext, mean=mean_rct-bias_ext, var=var_ext, corr=corr, beta_bias=beta_bias_ext)

# 2. Design
design = FixedRatioDesign(treat_ratio_prior=0.5)
split_data = design.split(rct_data, ext_data)

X_external = split_data.X_external
X_control_int = split_data.X_control_int
X_treat = split_data.X_treat
Y_control_int = split_data.Y_control_int
Y_treat = split_data.Y_treat
Y_external = split_data.Y_external

# 3. Fit energy weighting estimator
estimator = EnergyWeightingEstimator()
estimation_result = estimator.estimate(split_data)
weights = estimation_result.weights_continuous / np.sum(estimation_result.weights_continuous)

# %%

# build prognostic model on external data
from sklearn.ensemble import RandomForestRegressor
prognostic_model = RandomForestRegressor()
prognostic_model.fit(X_external, Y_external)

# plot the predicted values of the prognostic model on the RCT data
prognostic_predictions = prognostic_model.predict(rct_data.X)
import matplotlib.pyplot as plt
plt.scatter(prognostic_predictions, rct_data.Y0, alpha=0.5)
plt.xlabel('Prognostic Model Predictions')
plt.ylabel('Actual Outcomes')
plt.title('Prognostic Model Predictions vs Actual Outcomes')
plt.show()


#%%

def compute_energy_terms(X1, X2):

    # Compute pairwise distances
    d_12 = cdist(X1, X2, p=2)
    d_11 = cdist(X1, X1, p=2)
    d_22 = cdist(X2, X2, p=2)

    # Compute terms
    term_cross = torch.mean(d_12)
    term_self_1 = torch.mean(d_11)
    term_self_2 = torch.mean(d_22)

    energy = 2 * term_cross - term_self_1 - term_self_2

    return energy, term_cross, term_self_1, term_self_2

#%%

K_samples = 500

# for each sample, compute some metrics and store them in a list
metrics_dict = {
    'energy': [],
    'energy_term1': [],
    'energy_term2': [],
    'energy_term3': [],
    'error': [],
    'energy_prognostic': [],
    'mean_prognostic': []
}

prog_RCT_control = prognostic_model.predict(X_control_int)
prog_RCT_ext = prognostic_model.predict(X_external)

for i in range(K_samples):

    indices = np.random.choice(len(weights), p=weights, size=30, replace=False)
    selected_ext_data = X_external[indices]

    pooled_controls = np.vstack((X_control_int, selected_ext_data))

    energy, term1, term2, term3 = compute_energy_terms(torch.tensor(pooled_controls, dtype=torch.float32), torch.tensor(X_treat, dtype=torch.float32))
    metrics_dict['energy'].append(energy)
    metrics_dict['energy_term1'].append(term1)
    metrics_dict['energy_term2'].append(term2)
    metrics_dict['energy_term3'].append(term3)

    ate_estimate = np.mean(Y_treat) - (np.sum(Y_control_int) + np.sum(Y_external[indices])) / (len(Y_control_int) + len(indices))
    error = np.abs(ate_estimate - treatment_effect)
    metrics_dict['error'].append(error)

    energy_p, _, _, _ = compute_energy_terms(torch.tensor(prog_RCT_control).unsqueeze(1), torch.tensor(prog_RCT_ext[indices]).unsqueeze(1))
    metrics_dict['energy_prognostic'].append(energy_p)
    
    mean_p = np.abs(np.mean(prog_RCT_control) - np.mean( prog_RCT_ext[indices]))
    metrics_dict['mean_prognostic'].append(mean_p)


#%%

#%%

import matplotlib.pyplot as plt
import seaborn as sns

# Convert metrics to a DataFrame for analysis
# This ensures torch tensors are converted to floats and handles the list structure
results_list = []
for i in range(len(metrics_dict['error'])):
    row = {
        'error': metrics_dict['error'][i],
        'energy': metrics_dict['energy'][i].item() if torch.is_tensor(metrics_dict['energy'][i]) else metrics_dict['energy'][i],
        'energy_term1': metrics_dict['energy_term1'][i].item() if torch.is_tensor(metrics_dict['energy_term1'][i]) else metrics_dict['energy_term1'][i],
        'energy_prognostic': metrics_dict['energy_prognostic'][i].item() if torch.is_tensor(metrics_dict['energy_prognostic'][i]) else metrics_dict['energy_prognostic'][i],
        'mean_prognostic': np.abs(metrics_dict['mean_prognostic'][i]) # Use absolute difference for bias investigation
    }
    results_list.append(row)

df_analysis = pd.DataFrame(results_list)

# 1. Plot Metric vs. Error Correlations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

# Define the metrics we want to visualize against the error
plot_specs = [
    ('energy', 'Total Energy (Covariates)'),
    ('energy_term1', 'Energy First Term (Cross-term)'),
    ('energy_prognostic', 'Energy (Prognostic Scores)'),
    ('mean_prognostic', 'Abs Mean Diff (Prognostic)')
]

for ax, (col, label) in zip(axes, plot_specs):
    # Scatter plot with a linear regression fit
    sns.regplot(data=df_analysis, x=col, y='error', ax=ax, 
                scatter_kws={'alpha':0.5, 'color': 'steelblue'}, 
                line_kws={'color': 'darkred'})
    
    # Calculate Pearson correlation
    corr = df_analysis[col].corr(df_analysis['error'])
    
    ax.set_title(f"{label} vs ATE Error\nCorrelation: {corr:.3f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Absolute ATE Error")

plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
# This helps see if metrics are redundant or provide unique information
plt.figure(figsize=(10, 8))
correlation_matrix = df_analysis.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.title("Correlation Matrix: Metrics and ATE Error")
plt.show()

# 3. Scatter plot: energy cov, energy prog, colour=Error
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_analysis['energy'], df_analysis['energy_prognostic'], 
                      c=df_analysis['error'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Absolute ATE Error')
plt.xlabel('Total Energy (Covariates)')
plt.ylabel('Energy (Prognostic Scores)')
plt.title('Energy (Covariates) vs Energy (Prognostic) Colored by ATE Error')
plt.show()