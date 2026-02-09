#%%

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from estimator import PrognosticEnergyWeightingEstimator

#%%

def generate_synthetic_data(n_samples=1000, n_features=5):
    X = np.random.uniform(-2, 2, (n_samples, n_features))
    
    # Non-linear prognostic function (ground truth)
    # y = 2*x0 + sin(x1 * pi) + 0.5*x2^2 + noise
    m_x = (2 * X[:, 0] + 
           np.sin(X[:, 1] * np.pi) + 
           0.5 * X[:, 2]**2)
    
    y = m_x + np.random.normal(0, 0.5, n_samples)
    return X, y, m_x

def test_prognostic_quality():
    # 1. Setup
    n_features = 5
    device = torch.device("cpu")
    est = PrognosticEnergyWeightingEstimator(device=device)
    
    # 2. Generate Data
    X_train_np, Y_train_np, _ = generate_synthetic_data(1200, n_features)
    X_test_np, Y_test_np, m_x_true = generate_synthetic_data(400, n_features)
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    
    print(f"Training Prognostic Model on {len(X_train)} samples...")
    
    # 3. Train using the internal method
    # This uses the Random Forest logic you implemented
    prog_model = est._train_prognostic_model(X_train, Y_train)
    
    # 4. Predict
    with torch.no_grad():
        m_x_pred = prog_model(X_test).numpy().flatten()
    
    # 5. Metrics
    mse = mean_squared_error(m_x_true, m_x_pred)
    r2 = r2_score(m_x_true, m_x_pred)
    
    print("\n--- Prognostic Model Evaluation ---")
    print(f"MSE (vs True m(x)): {mse:.4f}")
    print(f"R^2 Score:          {r2:.4f}")
    
    # 6. Quick Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(m_x_true, m_x_pred, alpha=0.5, color='teal')
    plt.plot([m_x_true.min(), m_x_true.max()], [m_x_true.min(), m_x_true.max()], 'r--')
    plt.xlabel("True Prognostic Score (m(x))")
    plt.ylabel("Predicted Prognostic Score")
    plt.title("Prognostic Model: Truth vs Prediction")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    test_prognostic_quality()
# %%
