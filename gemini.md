# Project: Energy-Based Data Fusion for RCTs
Owner: Christophe (PhD, Oxford)

## Overview
We are developing a method to augment RCT control arms with external control data. 
The core problem is distributional mismatch between external controls and the trial population.
We solve this using Energy Statistics (Energy Distance) to optimize weights or select a representative sample.

## Mathematical Context
- **Target Distribution (Q):** RCT Treatment arm.
- **Source Distribution (P):** Pooled Control (RCT Control + External).
- **Objective:** Minimize Energy Distance $\mathcal{E}(P, Q)$ to find weights $w$.
- **Refinement:** Use optimal weights $w$ to define a sampling probability, then select the "Best-of-K" discrete cohort that minimizes empirical Energy Distance.

## Current Status
- Basic simulation implemented in Python (numpy/scipy).
- 3 groups: RCT Treat, RCT Control, External.
- Model implemented: Linear outcome, Normal covariates.
- More complexity to be added (non-normal cov, non-normal shift, ...) 

## Constraints
- **Code Style:** No comments in code.
- **Tools:** standard python stack (numpy, scipy).