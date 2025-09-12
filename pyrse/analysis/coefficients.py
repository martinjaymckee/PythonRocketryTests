from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict
import numpy as np

from typing import List, Dict, Tuple, Optional

class BaseRegressor:
    def __init__(self):
        self.param_names = {}
        self.param_stats = {}
        self.residuals = []
        self.goodness_of_fit = {}

    def fit(self, samples: List['CoefficientSample']):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __call__(self, **params):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def combine(self, samples: List['CoefficientSample'], regressor_cls=None):
        if regressor_cls is None:
            regressor_cls = self.__class__
        new_regressor = regressor_cls()
        new_regressor.fit(samples)
        return CompoundRegressor([self, new_regressor], mode='parallel')
    
    def chain(self, samples: List['CoefficientSample'], regressor_cls=None):
        if regressor_cls is None:
            regressor_cls = self.__class__
        new_regressor = regressor_cls()
        new_regressor.fit(samples)
        return CompoundRegressor([self, new_regressor], mode='sequential')
    

class CompoundRegressor:
    def __init__(self, regressors: List[BaseRegressor], mode: str = 'parallel'):
        if mode not in ('parallel', 'sequential'):
            raise ValueError("Mode must be 'parallel' or 'sequential'.")
        super().__init__()
        self.regressors = regressors
        self.mode = mode

    def fit(self, samples: List['CoefficientSample'] ->Tuple[List[float], Dict]):
        residuals = []
        for reg in self.regressors:
            res, _ = reg.fit(samples)
            residuals.extend(res)
        self.residuals = residuals
        self.goodness_of_fit = {'compound_mode': self.mode, 'n_regressors': len(self.regressors)}
        return self.residuals, self.goodness_of_fit

    def __call__(self, **params):
        results = [reg(**params) for reg in self.regressors]
        if self.mode == 'parallel':
            values, uncerts = zip(*results)
            # TODO: THESE CALCULATIONS NEED TO BE WEIGHTED
            mean_val = float(np.mean(np.array(values))) 
            combined_uncert = float(np.sqrt(np.mean(np.array(uncerts)**2)))
            return mean_val, combined_uncert
        elif self.mode == 'sequential':
            val, uncert = results[0]
            for reg in self.regressors[1:]:
                if isinstance(reg, CompoundRegressor):
                    val, uncert = reg**{**params, 'input': val}
                return val, uncert
        else:
            raise ValueError(f'Unknown Mode {self.mode}')

    def tree_view(self, indent: int = 0) -> str:
        pad = '  ' * indent
        desc = f'{pad}{self.__class__.__name__} (mode={self.mode})\n'
        for reg in self.regressors:
            if isinstance(reg, CompoundRegressor):
                desc += reg.tree_view(indent + 1)
            else:
                desc += f'{pad}  {reg.__class__.__name__}\n'
        return desc           
    
    
class ParameterStatistics:
    def __init__(self, samples):
        if not samples:
            raise ValueError("Samples list cannot be empty.")
        self.param_keys = set(samples[0].parameters.keys())
        for sample in samples:
            if set(sample.parameters.keys()) != self.param_keys:
                raise ValueError("All samples must have the same parameter keys.")
        self.stats = self._compute_stats(samples)

    def _compute_stats(self, samples):
        stats = {}
        for key in self.param_keys:
            values = np.array([sample.parameters[key] for sample in samples])
            mean = np.mean(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            stddev = np.std(values)
            stats[key] = {
                'mean': mean,
                'stddev': stddev,
                'min': np.min(values),
                'max': np.max(values)
            }
        return stats

    def mean(self, param):
        return self.stats[param]['mean']

    def stddev(self, param):
        return self.stats[param]['stddev']

    def min(self, param):
        return self.stats[param]['min']

    def max(self, param):
        return self.stats[param]['max']

    def summary(self):
        return self.stats


@dataclass
class CoefficientSample:
    coefficient: float
    parameters: Dict[str, float]
    weight: float


class CoefficientMapping:
    def __init__(self, samples, regressor=None):
        """
        Wraps a set of CoefficientSample objects and a regressor value to a coefficient value and an uncertainty estimate.

        :param samples: list of CoefficientSample objects
        :param regressor: str, class, or None, the regressor type associated with this mapping
        """
        self._validate_samples(samples)

        self.samples = samples
        self.regressor = regressor

        # Example: Compute coefficient and uncertainty from samples
        self.coefficient = self._compute_coefficient()
        self.uncertainty = self._compute_uncertainty()

    def _validate_samples(self, samples):
        if not samples:
            raise ValueError("Samples list cannot be empty.")
        param_keys = None
        for sample in samples:
            if sample.weight < 0:
                raise ValueError("Sample weights must be non-negative.")
            if param_keys is None:
                param_keys = set(sample.parameters.keys())
            elif set(sample.parameters.keys()) != param_keys:
                raise ValueError("All samples must have the same parameter keys.")
            
    def _compute_coefficient(self):
        # Placeholder: mean of sample coefficients
        return sum(sample.coefficient for sample in self.samples) / len(self.samples)

    def _compute_uncertainty(self):
        # Placeholder: standard deviation of sample coefficients
        mean = self.coefficient
        variance = sum((sample.coefficient - mean) ** 2 for sample in self.samples) / len(self.samples)
        return variance ** 0.5

    def __repr__(self):
        return (f"CoefficientMapping(regressor={self.regressor}, "
            f"coefficient={self.coefficient}, uncertainty={self.uncertainty})")