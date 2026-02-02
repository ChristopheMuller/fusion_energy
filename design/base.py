from abc import ABC, abstractmethod
from typing import Optional
from structures import PotentialOutcomes, SplitData
import numpy as np

class BaseDesign(ABC):
    """
    Abstract base class for design strategies.
    
    A design strategy determines how to split the available RCT pool
    (and potentially how much external data to include) before
    any estimation or outcome observation takes place.
    """
    
    @abstractmethod
    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes, rng: Optional[np.random.Generator] = None) -> SplitData:
        """
        Splits the RCT pool into Treatment and Internal Control, and determines
        the target augmentation size.
        
        Args:
            rct_pool: PotentialOutcomes object containing RCT covariates and hidden outcomes.
            ext_pool: PotentialOutcomes object containing External covariates and outcomes.
            rng: Optional numpy Generator for reproducible splitting.
            
        Returns:
            SplitData: Object containing the specific subsets to be used for estimation.
        """
        pass