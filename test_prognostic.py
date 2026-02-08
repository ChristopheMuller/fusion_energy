#%%

import numpy as np
from structures import EstimationResult
from generators import DataGenerator


#%% GENERATE DATA 

np.random.seed(1234)

dim = 2

beta = np.ones(dim)

n_rct = 1000
n_ext = 5000

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
for epoch in range(100):
    prog_opt.zero_grad()
    
    # We sample k=1 for both predictions to compute the standard Energy Score 
    # Output shape will be [batch_size, 1, 1]
    y_pred1 = prog_model(X_ext_t)
    y_pred2 = prog_model(X_ext_t)

    # Energy score: E|y_pred - y| - 0.5 * E|y_pred1 - y_pred2| 
    # Squeeze the k and output dimensions for the mean calculation
    loss_es = torch.mean(torch.abs(y_pred1.squeeze() - Y_ext_t.squeeze())) - \
              0.5 * torch.mean(torch.abs(y_pred1.squeeze() - y_pred2.squeeze()))
    
    loss_es.backward()
    prog_opt.step()

prog_model.eval()


#%% Evaluate prognostic model (plot predicted vs actual for E1)

with torch.no_grad():

    y_pred_e1 = prog_model(X_ext_t).reshape(X_ext_t.shape[0], -1).cpu().numpy().mean(axis=1)

plt.figure(figsize=(6, 6))
plt.scatter(Y_ext_t.cpu().numpy(), y_pred_e1, alpha=0.5)
plt.plot([Y_ext_t.min(), Y_ext_t.max()], [Y_ext_t.min(), Y_ext_t.max()], 'r--')
plt.xlabel('Actual Y (E1)')
plt.ylabel('Predicted Y (E1)')
plt.title('Prognostic Model Fit on E1')
plt.show()


#%% Optimize weights w

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


Cov_trt_tensor = torch.tensor(X_treat, dtype=torch.float32, device=device)
Cov_ctrl_tensor = torch.tensor(X_control, dtype=torch.float32, device=device)
Cov_ext_tensor = torch.tensor(X_ext, dtype=torch.float32, device=device)
Cov_ctrlPooled_tensor = torch.cat([Cov_ctrl_tensor, Cov_ext_tensor], dim=0)

with torch.no_grad():
    Outcome_Ext = torch.tensor(prog_model(X_ext_t).reshape(X_ext_t.shape[0], -1).cpu().numpy().mean(axis=1), dtype=torch.float32, device=device).view(-1, 1)
    Outcome_Rct_control = torch.tensor(prog_model(X_control_t).reshape(X_control_t.shape[0], -1).cpu().numpy().mean(axis=1), dtype=torch.float32, device=device).view(-1, 1)

#%%

# Loss: ED(Q, P^w) + ED( w*m(X_ext), w*m(X_rct))

lamb = 0.1


logits = torch.zeros(n_ext + Cov_ctrl_tensor.shape[0], requires_grad=True, device=device)
opt = torch.optim.Adam([logits], lr=0.01)

for epoch in range(500):

    opt.zero_grad()
    
    w = F.softmax(logits, dim=0)
    w_rct_control = w[:Cov_ctrl_tensor.shape[0]]
    w_ext = w[Cov_ctrl_tensor.shape[0]:]

    loss1 = energy_distance(Cov_trt_tensor, Cov_ctrlPooled_tensor, None, torch.cat([w_rct_control, w_ext], dim=0))
    loss2 = energy_distance(Outcome_Rct_control, Outcome_Ext, None, w_ext)
    loss = loss1 + lamb * loss2

    loss.backward()
    opt.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}")
# %%

import numpy as np
import matplotlib.pyplot as plt

def to_np(x):
    return x.detach().cpu().numpy()

def weighted_mean(x, w):
    # x: [n, d] or [n,]
    x = np.asarray(x)
    w = np.asarray(w)
    w = w / w.sum()
    return (w[:, None] * x).sum(axis=0) if x.ndim == 2 else (w * x).sum()

def weighted_var(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    w = w / w.sum()
    mu = weighted_mean(x, w)
    if x.ndim == 2:
        return (w[:, None] * (x - mu)**2).sum(axis=0)
    else:
        return (w * (x - mu)**2).sum()

def ess(w):
    w = np.asarray(w)
    w = w / w.sum()
    return 1.0 / np.sum(w**2)

def weighted_hist(ax, x, w=None, bins=40, density=True, alpha=0.4, label=None):
    x = np.asarray(x).ravel()
    if w is None:
        ax.hist(x, bins=bins, density=density, alpha=alpha, label=label)
    else:
        w = np.asarray(w).ravel()
        w = w / w.sum()
        ax.hist(x, bins=bins, weights=w, density=density, alpha=alpha, label=label)

def plot_covariates_weighted(X_treat, X_control, X_ext, w_rct_control, w_ext, title_suffix=""):
    """
    Shows:
      - scatter of X (treated, control, external)
      - marginal hist for each dimension with weighted pooled controls
    """
    w_r = np.asarray(w_rct_control); w_e = np.asarray(w_ext)
    w_pool = np.concatenate([w_r, w_e])
    X_pool = np.vstack([X_control, X_ext])

    d = X_treat.shape[1]
    fig = plt.figure(figsize=(5*d, 8))

    # --- Scatter for d=2 (or show first 2 dims)
    ax0 = plt.subplot2grid((2, d), (0, 0), colspan=d)
    if d >= 2:
        ax0.scatter(X_control[:, 0], X_control[:, 1], alpha=0.35, label="RCT control (raw)")
        ax0.scatter(X_treat[:, 0], X_treat[:, 1], alpha=0.35, label="RCT treated")
        ax0.scatter(X_ext[:, 0], X_ext[:, 1], s=5, alpha=0.12, label="External (raw)")
        # weighted external as bigger points (sample proportional to weight rank)
        idx = np.argsort(w_e)[-min(150, len(w_e)):]  # top weights
        ax0.scatter(X_ext[idx, 0], X_ext[idx, 1], s=30, alpha=0.7, label="External (top weights)")
        ax0.set_xlabel("X1"); ax0.set_ylabel("X2")
        ax0.set_title(f"Covariate scatter (top-weighted external highlighted){title_suffix}")
        ax0.legend()
    else:
        ax0.axis("off")

    # --- Marginal distributions per dim
    for j in range(d):
        ax = plt.subplot2grid((2, d), (1, j))
        weighted_hist(ax, X_treat[:, j], None, label="Treated", alpha=0.35)
        weighted_hist(ax, X_control[:, j], None, label="RCT ctrl raw", alpha=0.35)
        weighted_hist(ax, X_pool[:, j], w_pool, label="Pooled ctrl weighted", alpha=0.35)
        weighted_hist(ax, X_ext[:, j], None, label="External raw", alpha=0.20)
        ax.set_title(f"X{j+1} marginals")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_outcomes_and_prognosis(Y_control, Y_treat, Y_ext, m_control, m_treat, m_ext, w_ext, title_suffix=""):
    """
    Outcome Y:
      - RCT control raw
      - RCT treated raw
      - external raw
      - external weighted (weights only apply to external here)

    Prognostic m(X):
      - RCT control raw
      - RCT treated raw
      - external raw
      - external weighted
    """
    w_e = np.asarray(w_ext)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Outcomes
    ax = axes[0]
    weighted_hist(ax, Y_control, None, label="RCT ctrl Y0", alpha=0.35)
    weighted_hist(ax, Y_treat, None, label="RCT treated Y1", alpha=0.35)
    weighted_hist(ax, Y_ext, None, label="External Y0 raw", alpha=0.20)
    weighted_hist(ax, Y_ext, w_e, label="External Y0 weighted", alpha=0.35)
    ax.set_title(f"Outcome distributions{title_suffix}")
    ax.legend()

    # Prognostic score
    ax = axes[1]
    weighted_hist(ax, m_control, None, label="RCT ctrl m(X)", alpha=0.35)
    weighted_hist(ax, m_treat, None, label="RCT treated m(X)", alpha=0.35)
    weighted_hist(ax, m_ext, None, label="External m(X) raw", alpha=0.20)
    weighted_hist(ax, m_ext, w_e, label="External m(X) weighted", alpha=0.35)
    ax.set_title(f"Prognostic score distributions{title_suffix}")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_lambda_sweep(lambdas, metrics, title="Lambda sweep"):
    """
    metrics: dict with keys -> list same length as lambdas
      suggested keys:
        - 'loss1', 'loss2', 'loss'
        - 'ess_ext'
        - 'prog_mean_gap'  (mean m_control - mean m_ext_weighted)
        - 'cov_mean_l1'    (L1 distance between treated mean and weighted pooled control mean)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(lambdas, metrics["loss1"], marker="o", label="loss1 (cov ED)")
    ax.plot(lambdas, metrics["loss2"], marker="o", label="loss2 (prog ED)")
    ax.plot(lambdas, metrics["loss"],  marker="o", label="total")
    ax.set_title("Objective components")
    ax.set_xlabel("lambda")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(lambdas, metrics["ess_ext"], marker="o")
    ax.set_title("External ESS (higher = less peaky)")
    ax.set_xlabel("lambda")

    ax = axes[1, 0]
    ax.plot(lambdas, metrics["prog_mean_gap"], marker="o")
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Prognostic mean gap: E[m|RCT ctrl] - E_w[m|ext]")
    ax.set_xlabel("lambda")

    ax = axes[1, 1]
    ax.plot(lambdas, metrics["cov_mean_l1"], marker="o")
    ax.set_title("Mean-covariate L1 gap: treated vs weighted pooled control")
    ax.set_xlabel("lambda")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def ate_weighted_pool(Y_treat, Y_control, Y_ext, w_rct_control, w_ext):
    """
    Treat group: unweighted mean of Y_treat
    Counterfactual control mean: weighted mean over pooled controls
    """
    mu1 = np.mean(Y_treat)
    mu0 = np.sum(w_rct_control * Y_control) + np.sum(w_ext * Y_ext)
    return mu1 - mu0


#%%

with torch.no_grad():
    # weights
    w = F.softmax(logits, dim=0)
    w_rct_control = to_np(w[:Cov_ctrl_tensor.shape[0]])
    w_ext = to_np(w[Cov_ctrl_tensor.shape[0]:])

    # prognostic scores
    m_control = to_np(prog_model(X_control_t)).ravel()
    m_treat   = to_np(prog_model(X_treat_t)).ravel()
    m_ext     = to_np(prog_model(X_ext_t)).ravel()

# --- covariates (weighted)
plot_covariates_weighted(
    X_treat=X_treat,
    X_control=X_control,
    X_ext=X_ext,
    w_rct_control=w_rct_control,
    w_ext=w_ext,
    title_suffix=f"  (lambda={lamb})"
)

# --- outcomes + prognosis (weights applied to external only; that’s the only place weights “make sense”)
plot_outcomes_and_prognosis(
    Y_control=Y_control,
    Y_treat=Y_treat,
    Y_ext=Y_ext,
    m_control=m_control,
    m_treat=m_treat,
    m_ext=m_ext,
    w_ext=w_ext,
    title_suffix=f"  (lambda={lamb})"
)

print("ESS external:", ess(w_ext))
print("ESS pooled :", ess(np.concatenate([w_rct_control, w_ext])))

#%%

lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

metrics = {"loss1": [], "loss2": [], "loss": [], "ess_ext": [],
           "prog_mean_gap": [], "cov_mean_l1": [], "ate_hat": []}

for lamb in lambdas:
    logits = torch.zeros(n_ext + Cov_ctrl_tensor.shape[0], requires_grad=True, device=device)
    opt = torch.optim.Adam([logits], lr=0.01)

    for epoch in range(100):
        opt.zero_grad()

        w = F.softmax(logits, dim=0)
        w_rct_control = w[:Cov_ctrl_tensor.shape[0]]
        w_ext = w[Cov_ctrl_tensor.shape[0]:]

        w_pool = torch.cat([w_rct_control, w_ext], dim=0)

        loss1 = energy_distance(Cov_trt_tensor, Cov_ctrlPooled_tensor, None, w_pool)
        loss2 = energy_distance(Outcome_Rct_control, Outcome_Ext, None, w_ext)
        loss = loss1 + lamb * loss2

        loss.backward()
        opt.step()

    # --- record metrics
    with torch.no_grad():
        w = F.softmax(logits, dim=0)
        w_r = to_np(w[:Cov_ctrl_tensor.shape[0]])
        w_e = to_np(w[Cov_ctrl_tensor.shape[0]:])

        # prognostic mean gap
        m_control = to_np(prog_model(X_control_t)).ravel()
        m_ext = to_np(prog_model(X_ext_t)).ravel()
        prog_gap = m_control.mean() - weighted_mean(m_ext, w_e)

        # treated mean vs weighted pooled ctrl mean gap
        X_pool = np.vstack([X_control, X_ext])
        w_pool = np.concatenate([w_r, w_e])
        cov_gap = np.abs(X_treat.mean(axis=0) - weighted_mean(X_pool, w_pool)).sum()

        # recompute losses for reporting
        w_torch_pool = torch.tensor(w_pool, dtype=torch.float32, device=device)
        w_torch_ext  = torch.tensor(w_e, dtype=torch.float32, device=device)

        l1 = energy_distance(Cov_trt_tensor, Cov_ctrlPooled_tensor, None, w_torch_pool).item()
        l2 = energy_distance(Outcome_Rct_control, Outcome_Ext, None, w_torch_ext).item()
        lt = l1 + lamb * l2

        ate_hat = ate_weighted_pool(Y_treat, Y_control, Y_ext, w_r, w_e)

    metrics["loss1"].append(l1)
    metrics["loss2"].append(l2)
    metrics["loss"].append(lt)
    metrics["ess_ext"].append(ess(w_e))
    metrics["ate_hat"].append(ate_hat)
    metrics["prog_mean_gap"].append(prog_gap)
    metrics["cov_mean_l1"].append(cov_gap)

plot_lambda_sweep(lambdas, metrics, title="Effect of lambda on balance + weight concentration")

plt.figure(figsize=(7, 4))
plt.plot(lambdas, metrics["ate_hat"], marker="o", label="Estimated ATE (weighted)")
plt.axhline(2.0, linestyle="--", label="True ATE = 2.0")
plt.xlabel("lambda")
plt.ylabel("ATE estimate")
plt.title("Estimated ATE vs lambda")
plt.legend()
plt.tight_layout()
plt.show()
