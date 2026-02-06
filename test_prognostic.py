#%%

import numpy as np
from structures import EstimationResult
from generators import DataGenerator


#%% GENERATE DATA 

np.random.seed(1234)

dim = 2

beta = np.ones(dim)

n_rct = 2000
n_ext = 10000

mean_rct = np.ones(dim)
var_rct = 1.0
corr = 0.3

bias_ext = 0.7
var_ext = 1.5
beta_bias_ext = np.zeros(dim)

def cate_function(X):
    """Defines the Treatment Effect as a function of covariates X."""
    return 2.0 * np.ones(X.shape[0])
treatment_effect = cate_function


gen = DataGenerator(dim=dim, beta=beta)

rct_data = gen.generate_rct_pool(n=n_rct, mean=mean_rct, var=var_rct, corr=corr, treatment_effect=treatment_effect)
ext_data = gen.generate_external_pool(n=n_ext, mean=mean_rct-bias_ext, var=var_ext, corr=corr, beta_bias=beta_bias_ext)

shuffle_indices = np.random.permutation(n_rct)
X_control = rct_data.X[shuffle_indices[:n_rct // 2]]
Y_control = rct_data.Y0[shuffle_indices[:n_rct // 2]]
X_treat = rct_data.X[shuffle_indices[n_rct // 2:]]
Y_treat = rct_data.Y1[shuffle_indices[n_rct // 2:]]

X_ext = ext_data.X
Y_ext = ext_data.Y0

#%% PLOTS INIT

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_control[:, 0], X_control[:, 1], color='blue', alpha=0.5, label='Control')
plt.scatter(X_treat[:, 0], X_treat[:, 1], color='red', alpha=0.5, label='Treatment')
plt.scatter(X_ext[:, 0], X_ext[:, 1], color='green', alpha=0.1, s=3, label='External')

plt.title('Covariate Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()


# plot density of outcomes, per group
from scipy.stats import gaussian_kde
plt.subplot(1, 2, 2)
kde_control = gaussian_kde(Y_control)
kde_treat = gaussian_kde(Y_treat)
kde_ext = gaussian_kde(Y_ext)
y_range = np.linspace(min(Y_control.min(), Y_treat.min(), Y_ext.min()) - 1, max(Y_control.max(), Y_treat.max(), Y_ext.max()) + 1, 100)
plt.plot(y_range, kde_control(y_range), color='blue', label='Control')
plt.plot(y_range, kde_treat(y_range), color='red', label='Treatment')
plt.plot(y_range, kde_ext(y_range), color='green', label='External')
plt.title('Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()


#%% Compute weights w



import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_control_t = torch.tensor(X_control, dtype=torch.float32, device=device)
X_treat_t = torch.tensor(X_treat, dtype=torch.float32, device=device)
X_ext_t = torch.tensor(X_ext, dtype=torch.float32, device=device)
Y_ext_t = torch.tensor(Y_ext, dtype=torch.float32, device=device).view(-1, 1)

n_ext_total = X_ext.shape[0]
indices = torch.randperm(n_ext_total)
split_idx = n_ext_total // 2

X_e1, Y_e1 = X_ext_t[indices[:split_idx]], Y_ext_t[indices[:split_idx]]
X_e2, Y_e2 = X_ext_t[indices[split_idx:]], Y_ext_t[indices[split_idx:]]

class Prognostic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """
        Returns output of shape [batch_size, k, 1]
        """
        batch_size = x.shape[0]
        # Repeat covariates for each of the k draws                 
        out = self.net(torch.cat([x], dim=1)) # [batch_size * k, 1]
        
        # Reshape to ensure k has its own dimension
        return out.view(batch_size, 1)

prog_model = Prognostic(dim).to(device)
prog_opt = torch.optim.Adam(prog_model.parameters(), lr=0.01)

# Training loop for prognostic model (Engression / Energy Score loss) 
for epoch in range(500):
    prog_opt.zero_grad()
    
    # We sample k=1 for both predictions to compute the standard Energy Score 
    # Output shape will be [batch_size, 1, 1]
    y_pred1 = prog_model(X_e1)
    y_pred2 = prog_model(X_e1)

    # Energy score: E|y_pred - y| - 0.5 * E|y_pred1 - y_pred2| 
    # Squeeze the k and output dimensions for the mean calculation
    loss_es = torch.mean(torch.abs(y_pred1.squeeze() - Y_e1.squeeze())) - \
              0.5 * torch.mean(torch.abs(y_pred1.squeeze() - y_pred2.squeeze()))
    
    loss_es.backward()
    prog_opt.step()

prog_model.eval()


#%% Evaluate prognostic model (plot predicted vs actual for E1)

with torch.no_grad():

    y_pred_e1 = prog_model(X_e1).reshape(X_e1.shape[0], -1).cpu().numpy().mean(axis=1)

plt.figure(figsize=(6, 6))
plt.scatter(Y_e1.cpu().numpy(), y_pred_e1, alpha=0.5)
plt.plot([Y_e1.min(), Y_e1.max()], [Y_e1.min(), Y_e1.max()], 'r--')
plt.xlabel('Actual Y (E1)')
plt.ylabel('Predicted Y (E1)')
plt.title('Prognostic Model Fit on E1')
plt.show()


#%% Evaluate prognostic model (plot predicted vs actual for E2)

with torch.no_grad():

    y_pred_e2 = prog_model(X_e2).reshape(X_e2.shape[0], -1).cpu().numpy().mean(axis=1)

plt.figure(figsize=(6, 6))
plt.scatter(Y_e2.cpu().numpy(), y_pred_e2, alpha=0.5)
plt.plot([Y_e2.min(), Y_e2.max()], [Y_e2.min(), Y_e2.max()], 'r--')
plt.xlabel('Actual Y (E2)')
plt.ylabel('Predicted Y (E2)')
plt.title('Prognostic Model Fit on E2')
plt.show()

#%% Optimize weights w

# LOSS = ED(Q, P^w) + ED( w*m(X_e2), m(X_rct))

def energy_distance(X, Y, w1, w2):
    
    if w1 is None:
        w1 = torch.ones(X.shape[0], device=X.device) / X.shape[0]
    if w2 is None:
        w2 = torch.ones(Y.shape[0], device=Y.device) / Y.shape[0]

    # Compute pairwise distances
    d_XY = torch.cdist(X, Y, p=1)  # [n_X, n_Y]
    d_XX = torch.cdist(X, X, p=1)  # [n_X, n_X]
    d_YY = torch.cdist(Y, Y, p=1)  # [n_Y, n_Y]

    # Compute energy distance
    term1 = torch.sum(w1.unsqueeze(1) * w2.unsqueeze(0) * d_XY)
    term2 = 0.5 * torch.sum(w1.unsqueeze(1) * w1.unsqueeze(0) * d_XX)
    term3 = 0.5 * torch.sum(w2.unsqueeze(1) * w2.unsqueeze(0) * d_YY)

    return term1 - term2 - term3


logits = torch.rand(X_e2.shape[0], device=X_e2.device, requires_grad=True)
opt = torch.optim.Adam([logits], lr=0.01)
for epoch in range(500):
    w = torch.softmax(logits, dim=0)
    
    ed_loss = torch.min(w)
    
    ed_loss.backward()
    opt.step()
    
    with torch.no_grad():
        logits -= 0.01 * logits.grad
        logits.grad.zero_()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Energy Distance Loss: {ed_loss.item():.4f}")

print("Optimized weights w (first 10):", w[:10].detach().cpu().numpy())
print("max w:", w.max().item(), "min w:", w.min().item())
print("Sum of w:", w.sum().item())