from abc import ABC, abstractmethod
from structures import PotentialOutcomes, SplitData

class BaseDesign(ABC):
    """
    Abstract base class for design strategies.
    
    A design strategy determines how to split the available RCT pool
    (and potentially how much external data to include) before
    any estimation or outcome observation takes place.
    """
    
    @abstractmethod
    def split(self, rct_pool: PotentialOutcomes, ext_pool: PotentialOutcomes) -> SplitData:
        """
        Splits the RCT pool into Treatment and Internal Control, and determines
        the target augmentation size.
        
        Args:
            rct_pool: PotentialOutcomes object containing RCT covariates and hidden outcomes.
            ext_pool: PotentialOutcomes object containing External covariates and outcomes.
            
        Returns:
            SplitData: Object containing the specific subsets to be used for estimation.
        """
        pass