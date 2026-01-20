from abc import ABC, abstractmethod
import numpy as np

class BaseAugmenter(ABC):
    """
    Abstract base class for data fusion/augmentation methods.
    
    Enforces a consistent API for:
    - fit(X_target, X_internal, X_external)
    - sample(X_target, X_internal, X_external, Y_internal=None, Y_external=None)
    - get_internal_weights()
    """
    
    def __init__(self):
        self.weights_ = None

    @abstractmethod
    def fit(self, X_target, X_internal, X_external):
        """
        Fit the model to learn weights or parameters.
        Must set self.weights_.
        """
        pass

    @abstractmethod
    def sample(self, X_target, X_internal, X_external, Y_internal=None, Y_external=None):
        """
        Return the augmented dataset or weights.
        
        Returns:
            X_fused (np.ndarray or None): The combined/selected covariate data.
            Y_fused (np.ndarray or None): The combined/selected outcome data.
            weights (np.ndarray): Weights corresponding to the fused data or the original pool.
        """
        pass
    
    def get_internal_weights(self):
        """
        Returns the learned weights for the control pool.
        """
        return self.weights_
