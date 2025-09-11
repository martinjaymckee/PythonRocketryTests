from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict
import numpy as np

import pyrse.analysis.regression as regres


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