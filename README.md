# Energy-Based Data Fusion for RCTs

This project implements a simulation framework to evaluate methods for augmenting Randomized Controlled Trial (RCT) control arms with external data (e.g., Real-World Data). The core innovation is the use of **Energy Statistics** (specifically Energy Distance) to address distributional mismatches between the trial population and external data sources.

## Simulation Setup

The simulation compares different strategies for incorporating external control data to estimate the Average Treatment Effect (ATE).

### 1. Data Generation (`generators.py`)
Data is generated synthetically to mimic scenarios with covariate shift and potential concept drift.

*   **Covariates ($X$):** Generated from a Multivariate Normal distribution with a Toeplitz covariance structure (simulating correlated features).
*   **Outcomes ($Y$):** 
    *   Base outcome: $Y_0 = X\beta + \epsilon$, where $\epsilon \sim N(0, 0.5)$.
    *   Treatment effect: Added to $Y_0$ to get $Y_1$. The effect can be constant or a function of $X$ (CATE).
*   **Populations:**
    *   **RCT Pool:** Sampled from $N(\mu_{rct}, \Sigma)$. Represents the trial population.
    *   **External Pool:** Sampled from $N(\mu_{rct} - \text{bias}, \Sigma_{ext})$. Represents a biased external population (e.g., different demographics). We can also introduce "concept drift" by altering the $\beta$ coefficients for this group.

### 2. Experimental Design (`design/`)
This stage determines how the available RCT patients are allocated to Treatment and Control arms, and how many external patients are targeted for recruitment ($N_{aug}$).

*   **Fixed Ratio Design:** Standard randomization (e.g., 1:1). It can specify a fixed target number of external patients to recruit ($N_{aug}$).
*   **Energy Optimised Design:** A novel adaptive design.
    *   It uses **Cross-Validation** within the RCT data (splitting it into pseudo-target and pseudo-internal arms) to estimate the optimal number of external patients ($N_{aug}$) to add.
    *   The objective is to minimize the expected Energy Distance between the Treatment arm and the Augmented Control arm.

### 3. Estimation (`estimator/`)
Once the data is split, the estimator calculates the ATE. These estimators leverage the external data as follows:

*   **Matching Estimators (`EnergyMatchingEstimator`):** 
    *   Selects a discrete subset of $N_{aug}$ patients from the external pool.
    *   **Selection Criterion:** Minimizes the Energy Distance between the Treatment arm and the pooled Control arm (Internal + External).
    *   **Inference:** ATE is the difference between the mean of the Treatment arm and the mean of the pooled Control arm.
*   **Weighting Estimators (`EnergyWeightingEstimator`):**
    *   Instead of hard selection, it assigns continuous weights to *all* external patients.
    *   Weights are optimized via Gradient Descent to minimize Energy Distance.
    *   **Inference:** ATE is calculated using a weighted average for the control arm.

## Pipelines

The simulation (`main.py`) runs several pipelines (i.e. combinations of Design and Estimators) and repeat the process several times to evaluate the quality of the estimates (MSE). Some pipelines are:

*   **RCT_ONLY:** Baseline using only randomized data.
*   **EnergyMatching_X:** Augments with a fixed number ($X$) of external patients chosen via Energy Matching.
*   **Energy_Matching_EnergyOpt:** Automatically selects the optimal number of external patients using the `EnergyOptimisedDesign` and then performs Energy Matching.
*   **Energy_Weighting:** Uses the optimal size from the design phase but applies soft weighting instead of matching.

## Replication

### Using `uv` (Recommended)
This project is managed with `uv`, a fast Python package manager.

1.  **Install uv:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
3.  **Run the Simulation:**
    ```bash
    uv run main.py
    ```
