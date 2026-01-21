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
