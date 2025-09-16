from collections import defaultdict
from dataclasses import dataclass
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

import pyrse.analysis.regression as regres
from pyrse.analysis.coefficient_utils import CoefficientSample

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


class CoefficientMapping:
    @classmethod
    def FromFlightData(cls, fd, coeff, params, regressor=None):
        samples = []
        coeff_values = fd[coeff].values
        # TODO: USE THE INTERPOLATION MASK TO MODIFY SAMPLE WEIGHT
        param_values = {}
        for param in params:
            param_values[param] = fd[param].values
            # TODO: USE THE INTERPOLATION MASK TO MODIFY SAMPLE WEIGHT
        for idx in range(len(coeff_values)):
            val = coeff_values[idx]
            if math.isnan(val):
                continue
            sample_params = {}
            for p in params:
                param_val = param_values[p][idx]
                if math.isnan(param_val):
                    continue
                sample_params[p] = param_val
            samples.append(CoefficientSample(coeff_values[idx], sample_params))
        return CoefficientMapping(samples, regressor=regressor)
    
    def __init__(self, samples, regressor=None):
        """
        Wraps a set of CoefficientSample objects and a regressor value to a coefficient value and an uncertainty estimate.

        :param samples: list of CoefficientSample objects
        :param regressor: str, class, or None, the regressor type associated with this mapping
        """
        self._validate_samples(samples)

        self.samples = samples
        self.regressor = regressor if regressor is not None else regres.selectRegressor(self.samples)
        self.regressor.fit(self.samples)

    def __call__(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Evaluate the coefficient mapping at the given parameters.

        :param params: dict of parameter values
        :return: tuple of (coefficient value, uncertainty estimate)
        """
        value, uncert = self.regressor(**params)
        return value, uncert    
    
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
            
    def __repr__(self):
        return (f"CoefficientMapping(regressor={self.regressor}, "
            f"coefficient={self.coefficient}, uncertainty={self.uncertainty})")