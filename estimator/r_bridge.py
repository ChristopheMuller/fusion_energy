import numpy as np
import time
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    RPY2_AVAILABLE = True
except (ImportError, ValueError):
    RPY2_AVAILABLE = False
from estimator.base import BaseEstimator
from structures import SplitData, EstimationResult

_SOURCED_R_FILES = set()
if RPY2_AVAILABLE:
    NP_CONVERTER = robjects.default_converter + numpy2ri.converter

class REstimator(BaseEstimator):
    def __init__(self, r_script_path: str, r_func_name: str, n_external: int = None):
        if not RPY2_AVAILABLE:
            raise ImportError("rpy2 is not available or R is not installed correctly.")
        super().__init__(n_external=n_external)
        self.r_script_path = r_script_path
        self.r_func_name = r_func_name
        self._r_func = None

    @property
    def r_func(self):

        if self.r_script_path not in _SOURCED_R_FILES:
            robjects.r.source(self.r_script_path)
            _SOURCED_R_FILES.add(self.r_script_path)
        
        if self._r_func is None:
            self._r_func = robjects.globalenv[self.r_func_name]
        return self._r_func

    def estimate(self, data: SplitData, n_external: int = None) -> EstimationResult:

        start_time = time.time()

        if n_external is not None:
            n_ext = n_external
        elif self.n_external is not None:
            n_ext = self.n_external
        elif data.target_n_aug is not None:
            n_ext = data.target_n_aug
        else:
            n_ext = None
              
        with localconverter(robjects.default_converter + numpy2ri.converter):
            r_result = self.r_func(
                X_treat=data.X_treat,
                Y_treat=data.Y_treat,
                X_control_int=data.X_control_int,
                Y_control_int=data.Y_control_int,
                X_external=data.X_external,
                Y_external=data.Y_external,
                true_sate=data.true_sate,
                n_external=n_ext if n_ext is not None else robjects.NA_Logical
            )  

        return EstimationResult(
            ate_est=float(r_result[0][0]),
            bias=float(r_result[1][0]),
            weights_continuous=np.asarray(r_result[2]),
            weights_external=np.asarray(r_result[3]),
            energy_distance=float(r_result[4][0]) if len(r_result) > 4 else 0.0,
            estimation_time=time.time() - start_time
        )