from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np

class BaseRegressor(ABC):
    def __init__(self):
        self.param_names: Optional[List[str]] = None
        self.param_stats: Optional[str, Dict[str, float]] = {}
        self.residuals_: Optional[List[float]] = None
        self.goodness_of_fit_: Optional[Dict[str, float]] = None

    def _compute_param_stats(self, samples):
        stats = {}
        for name in self.param_names:
            values = np.array([sample.parameters[name] for sample in samples])
            mean = np.mean(values)
            # variance = sum((v - mean) ** 2 for v in values) / len(values)
            stddev = np.std(values)
            stats[name] = {
                'mean': mean,
                'stddev': stddev,
                'min': np.min(values),
                'max': np.max(values)
            }
        self.param_stats = stats
        self.param_names = list(stats.keys())

    def _fill_missing_params(self, params: Dict[str, float]):
        filled = dict(params)
        for pname in self.param_names:
            if pname not in self.param_stats:
                filled[pname] = self.param_stats[pname]['mean']
            else:
                raise ValueError(f"Parameter '{pname}' not found in statistics.")
        return filled
    
    @abstractmethod
    def fit(self, samples):
        pass

    @abstractmethod
    def chain(self, samples, regressor_type=None):
        pass

    @abstractmethod
    def combine_weighted(self, samples, regressor_type=None, weight=None):
        pass

    @abstractmethod
    def __call__(self, params: Dict[str, float]):
        pass