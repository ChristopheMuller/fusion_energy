import numpy as np

class NaivePooled:
    """
    Naive Estimator (a): Difference in means between Whole Treated and Whole Pooled Control (RCT Control + External).
    Does not perform any matching or weighting.
    """
    def __init__(self):
        pass

    def estimate(self, X_t, Y_t, X_i, Y_i, X_e, Y_e):
        """
        Returns:
            att (float): The estimated Average Treatment Effect on Treated.
            n_observed (int): The number of control units used (RCT Control + External).
        """
        # Estimator: Mean(Treated) - Mean(Pooled Control)
        mu_t = np.mean(Y_t)
        
        Y_control_pool = np.concatenate([Y_i, Y_e])
        mu_c = np.mean(Y_control_pool)
        
        att = mu_t - mu_c
        n_observed = len(Y_control_pool)
        
        return att, n_observed

class NaiveRCT:
    """
    Naive Estimator (b): Difference in means between Whole Treated and RCT Control only.
    Ignores External data.
    """
    def __init__(self):
        pass

    def estimate(self, X_t, Y_t, X_i, Y_i, X_e, Y_e):
        """
        Returns:
            att (float): The estimated Average Treatment Effect on Treated.
            n_observed (int): The number of control units used (RCT Control only).
        """
        # Estimator: Mean(Treated) - Mean(RCT Control)
        mu_t = np.mean(Y_t)
        mu_c = np.mean(Y_i)
        
        att = mu_t - mu_c
        n_observed = len(Y_i)
        
        return att, n_observed
