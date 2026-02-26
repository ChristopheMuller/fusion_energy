import numpy as np
import sys
import os
import torch

# Ensure we can import boot
sys.path.append(os.getcwd())

import boot
# Monkey patch for speed - verify reproducibility with small params
boot.B_ITERATIONS = 5
boot.N_RCT = 50
boot.N_EXT = 100

from boot import run_single_simulation

def main():
    print("Running reproducibility check on boot.py...")

    seed = 42

    # Run 1
    print(f"Run 1 (Seed {seed})...")
    res1 = run_single_simulation(seed=seed, n_jobs_inner=1)

    # Run 2
    print(f"Run 2 (Seed {seed})...")
    res2 = run_single_simulation(seed=seed, n_jobs_inner=1)

    # Compare
    keys = ["Estimate", "CI Lower", "CI Upper", "True ATE"]
    mismatches = []

    for key in keys:
        val1 = res1[key]
        val2 = res2[key]
        if not np.isclose(val1, val2, atol=1e-8):
            mismatches.append(f"{key}: {val1} != {val2}")

    if mismatches:
        print("FAIL: Results differ!")
        for m in mismatches:
            print(m)
        sys.exit(1)
    else:
        print("SUCCESS: Results are identical.")
        sys.exit(0)

if __name__ == "__main__":
    main()
