from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor as SKGaussianProcess
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import PolynomialFeatures

class BaseRegressor:
    def __init__(self):
        self.param_names: Optional[List[str]] = None
        self.param_stats: Optional[Dict[str, float]] = {}
        self.residuals: Optional[List[float]] = None
        self.goodness_of_fit: Optional[Dict[str, float]] = None

    @abstractmethod
    def fit(self, samples):
        pass

    @abstractmethod
    def update(self, samples):
        pass

    @abstractmethod
    def chain(self, samples, regressor_type=None):
        pass

    @abstractmethod
    def __call__(self, params: Dict[str, float]):
        pass

    def combine(self, samples, regressor_cls=None):
        if regressor_cls is None:
            regressor_cls = self.__class__
        new_regressor = regressor_cls()
        new_regressor.fit(samples)
        return CompoundRegressor([self, new_regressor], mode='parallel')
    
    def chain(self, samples, regressor_cls=None):
        if regressor_cls is None:
            regressor_cls = self.__class__
        new_regressor = regressor_cls()
        new_regressor.fit(samples)
        return CompoundRegressor([self, new_regressor], mode='sequential')
    
    def _compute_param_stats(self, samples):
        stats = {}
        param_names = list(samples[0].parameters.keys())
        for name in param_names:
            values = [sample.parameters[name] for sample in samples]
            stats[name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'stddev': np.std(values)
            }
        self.param_names = param_names
        self.param_stats = stats
        
    def _fill_missing_params(self, params: Dict[str, float]) -> Dict[str, float]:
        filled = dict(params)
        if self.param_stats is None:
            raise ValueError("Parameter statistics not computed. Fit the regressor first.")
        filled_params = params.copy()
        for name in self.param_names:
            if name not in filled_params:
                if name in self.param_stats:
                    filled[name] = self.param_stats[name]['mean']
                else:
                    raise ValueError(f"Parameter '{name}' not found in parameter statistics.")
        return filled
    

class CompoundRegressor:
    def __init__(self, regressors: List[BaseRegressor], mode: str = 'parallel'):
        if mode not in ('parallel', 'sequential'):
            raise ValueError("Mode must be 'parallel' or 'sequential'.")
        super().__init__()
        self.regressors = regressors
        self.mode = mode

    def fit(self, samples):
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


class KNearestNeighborRegressor(BaseRegressor):
    def __init__(self, k: int = None):
        """
        k: number of neighbors to use; if None, defaults to sqrt(n_samples) during fit
        """
        super().__init__()
        self.samples = []
        self.k = k  # can be None; will set dynamically at fit

    def fit(self, samples: List[Dict]) -> Tuple[List[float], Dict]:
        """
        Store samples and compute residuals against k-NN predictions.
        """
        self.samples = samples
        self._compute_param_stats(samples)

        # Determine default k if not set
        if self.k is None:
            self.k = max(1, int(np.sqrt(len(samples))))

        residuals = []
        for s in samples:
            pred, _ = self.__call__(**s.parameters)
            residuals.append(s.coefficient - pred)

        self.residuals_ = residuals
        rmse = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
        self.goodness_of_fit_ = {"rmse": rmse, "n_samples": len(samples)}
        return self.residuals_, self.goodness_of_fit_

    def __call__(self, **params) -> Tuple[float, float]:
        """
        Predict coefficient based on k-nearest neighbors with distance weighting.
        Returns (prediction, confidence interval).
        """
        if not self.samples:
            raise ValueError("Regressor not fitted yet.")

        params = self._fill_missing_params(params)
        x = np.array([params[p] for p in self.param_names])

        # Compute distances to all samples
        distances = []
        for sample in self.samples:
            s_x = np.array([sample.parameters[p] for p in self.param_names])
            dist = np.linalg.norm(x - s_x)
            distances.append(dist)

        distances = np.array(distances)
        neighbor_indices = np.argsort(distances)[:self.k]
        neighbor_values = np.array([self.samples[i].coefficient for i in neighbor_indices])
        neighbor_distances = distances[neighbor_indices]

        # Weight by inverse distance (add small epsilon to avoid div by zero)
        eps = 1e-8
        weights = 1 / (neighbor_distances + eps)
        weights /= np.sum(weights)

        prediction = float(np.sum(weights * neighbor_values))

        # Estimate uncertainty as weighted std of neighbors
        conf = float(np.sqrt(np.sum(weights * (neighbor_values - prediction) ** 2)))

        return prediction, conf

    def update(self, new_samples: List[Dict]) -> None:
        """
        Add new samples and recompute residuals/goodness-of-fit.
        """
        self.samples.extend(new_samples)
        self.fit(self.samples)


class PolynomialRegressor(BaseRegressor):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

        # State
        self.coeffs = None
        self.cov = None
        self.residuals = None
        self.goodness_of_fit = None

        # Sufficient statistics
        self.XT_W_X = None
        self.XT_W_y = None
        self.sum_weights = 0.0
        self.sum_sq_residuals = 0.0
        self.n = 0  # number of samples
        self.p = None  # number of parameters

    def fit(self, samples, weights=None):
        """Fit from scratch"""
        self._compute_param_stats(samples)

        X = self.poly.fit_transform(
            [[s.parameters[name] for name in self.param_names] for s in samples]
        )
        y = np.array([s.coefficient for s in samples])
        n, p = X.shape
        self.p = p

        if weights is None:
            weights = np.ones(n)
        else:
            weights = np.array(weights, dtype=float)

        # Weighted design
        W_sqrt = np.sqrt(weights)
        X_w = X * W_sqrt[:, None]
        y_w = y * W_sqrt

        # Sufficient statistics
        self.XT_W_X = X_w.T @ X_w
        self.XT_W_y = X_w.T @ y_w
        self.sum_weights = np.sum(weights)
        self.n = n

        # Solve
        self._solve(X, y, weights)
        return self.residuals, self.goodness_of_fit

    def update(self, samples, weights=None):
        """Incrementally add samples without refitting everything"""
        X_new = self.poly.transform(
            [[s.parameters[name] for s in self.param_names] for s in samples]
        )
        y_new = np.array([s.coefficient for s in samples])
        n_new = len(samples)

        if weights is None:
            weights = np.ones(n_new)
        else:
            weights = np.array(weights, dtype=float)

        W_sqrt = np.sqrt(weights)
        X_w_new = X_new * W_sqrt[:, None]
        y_w_new = y_new * W_sqrt

        # Update sufficient statistics
        self.XT_W_X += X_w_new.T @ X_w_new
        self.XT_W_y += X_w_new.T @ y_w_new
        self.sum_weights += np.sum(weights)
        self.n += n_new

        # Recompute coefficients using updated stats
        self._solve(np.vstack([X_new]), np.concatenate([y_new]), weights)

    def _solve(self, X, y, weights):
        # Solve least squares with updated XT_W_X / XT_W_y
        self.coeffs = np.linalg.solve(self.XT_W_X, self.XT_W_y)

        # Residuals on *newest* data
        y_pred = X @ self.coeffs
        residuals = y - y_pred
        self.residuals = residuals

        # Weighted residual variance
        sigma2 = np.sum(weights * residuals**2) / (self.n - self.p)

        # Covariance of coefficients
        self.cov = sigma2 * np.linalg.inv(self.XT_W_X)

        self.goodness_of_fit = {
            'rmse': float(np.sqrt(np.average(residuals**2, weights=weights))),
            'n_samples': self.n,
            'n_params': self.p
        }

    def __call__(self, **params):
        if self.coeffs is None:
            raise ValueError("Regressor not fitted yet.")
        params = self._fill_missing_params(params)
        row = self.poly.transform([[params[name] for name in self.param_names]])[0]

        value = float(self.coeffs @ row)
        if self.cov is not None:
            variance = row @ self.cov @ row.T
            uncert = float(np.sqrt(max(variance, 0.0)))
        else:
            uncert = 0.0
        return value, uncert


import numpy as np
from sklearn.kernel_ridge import KernelRidge
from typing import List, Dict, Tuple
from copy import deepcopy

class RBFRegressor(BaseRegressor):
    def __init__(self, alpha=1e-5, gamma=None, n_bootstrap=5, random_seed=42):
        """
        alpha: regularization to prevent overfitting
        gamma: kernel width; if None, defaults to 1.0
        n_bootstrap: number of bootstrap resamples for uncertainty estimation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.model = None
        self.bootstrap_models = []
        self._X_min = None
        self._X_range = None

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self._X_min) / (self._X_range + 1e-12)

    def fit(self, samples: List[Dict]) -> Tuple[List[float], Dict]:
        self._compute_param_stats(samples)
        X = np.array([[s.parameters[name] for name in self.param_names] for s in samples])
        y = np.array([s.coefficient for s in samples])

        # Save min/range for normalization
        self._X_min = X.min(axis=0)
        self._X_range = X.max(axis=0) - self._X_min
        X_norm = self._normalize_X(X)

        if self.gamma is None:
            self.gamma = 2.0#1.0  # default smoothness

        # Fit main model
        self.model = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=self.gamma)
        self.model.fit(X_norm, y)

        # Compute residuals
        y_pred = self.model.predict(X_norm)
        self.residuals = (y - y_pred).tolist()
        self.goodness_of_fit = {
            "rmse": float(np.sqrt(np.mean((y - y_pred) ** 2))),
            "n_samples": len(samples),
        }

        # Bootstrap models for uncertainty estimation
        rng = np.random.default_rng(self.random_seed)
        self.bootstrap_models = []
        for _ in range(self.n_bootstrap):
            indices = rng.integers(0, len(samples), size=len(samples))
            X_bs, y_bs = X_norm[indices], y[indices]
            model_bs = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=self.gamma)
            model_bs.fit(X_bs, y_bs)
            self.bootstrap_models.append(model_bs)

        return self.residuals, self.goodness_of_fit

    def update(self, samples: List[Dict]) -> None:
        # KernelRidge does not support incremental updates; just refit
        self.fit(samples)

    def __call__(self, **params) -> Tuple[float, float]:
        if self.model is None:
            raise ValueError("Regressor not fitted yet.")
        params = self._fill_missing_params(params)
        X = np.array([[params[name] for name in self.param_names]])
        X_norm = self._normalize_X(X)

        # Predict main value
        val = float(self.model.predict(X_norm)[0])

        # Bootstrap uncertainty: standard deviation of predictions across bootstrap models
        if self.bootstrap_models:
            preds = [m.predict(X_norm)[0] for m in self.bootstrap_models]
            conf = float(np.std(preds))
        else:
            conf = float(np.std(self.residuals)) if self.residuals else 0.0

        return val, conf


# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor as SKGP
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from typing import List, Dict, Tuple

# class GaussianProcessRegressor(BaseRegressor):
#     def __init__(self, alpha=1e-4, n_bootstrap=50, random_seed=42, n_restarts_optimizer=5):
#         """
#         alpha: observation noise (fixed)
#         n_bootstrap: number of bootstrap models for uncertainty
#         random_seed: reproducibility
#         n_restarts_optimizer: number of restarts for hyperparameter optimization
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.n_bootstrap = n_bootstrap
#         self.random_seed = random_seed
#         self.n_restarts_optimizer = n_restarts_optimizer
#         self.model = None
#         self.bootstrap_models = []
#         self._X_min = None
#         self._X_range = None

#     def _normalize_X(self, X: np.ndarray) -> np.ndarray:
#         return (X - self._X_min) / (self._X_range + 1e-12)

#     def fit(self, samples: List[Dict]) -> Tuple[List[float], Dict]:
#         self._compute_param_stats(samples)
#         X = np.array([[s.parameters[name] for name in self.param_names] for s in samples])
#         y = np.array([s.coefficient for s in samples])

#         # Normalize inputs
#         self._X_min = X.min(axis=0)
#         self._X_range = np.ptp(X, axis=0)
#         X_norm = self._normalize_X(X)

#         # Data-driven length scale bounds: 0.01–1.0 fraction of parameter spread
#         length_scale_bounds = [(0.01, 1.0) for _ in self.param_names]
#         length_scale_init = np.array([0.1 for _ in self.param_names])  # initial guess

#         # Data-driven signal variance bounds based on target RMS
#         y_rms = np.std(y)
#         kernel = C(y_rms ** 2, (y_rms**2 * 0.01, y_rms**2 * 100)) * \
#                  RBF(length_scale=length_scale_init, length_scale_bounds=length_scale_bounds)

#         # Fit GP with hyperparameter optimization
#         self.model = SKGP(kernel=kernel, alpha=self.alpha, normalize_y=True,
#                           optimizer='fmin_l_bfgs_b', n_restarts_optimizer=self.n_restarts_optimizer)
#         self.model.fit(X_norm, y)

#         # Compute residuals
#         y_pred = self.model.predict(X_norm)
#         self.residuals = (y - y_pred).tolist()
#         self.goodness_of_fit = {
#             "rmse": float(np.sqrt(np.mean((y - y_pred) ** 2))),
#             "n_samples": len(samples),
#         }

#         # Bootstrap models for uncertainty estimation
#         rng = np.random.default_rng(self.random_seed)
#         self.bootstrap_models = []
#         for _ in range(self.n_bootstrap):
#             indices = rng.integers(0, len(samples), size=len(samples))
#             X_bs, y_bs = X_norm[indices], y[indices]
#             model_bs = SKGP(kernel=kernel, alpha=self.alpha, normalize_y=True)
#             model_bs.fit(X_bs, y_bs)
#             self.bootstrap_models.append(model_bs)

#         return self.residuals, self.goodness_of_fit

#     def update(self, samples: List[Dict]) -> None:
#         # GP cannot update incrementally; just refit
#         self.fit(samples)

#     def __call__(self, **params) -> Tuple[float, float]:
#         if self.model is None:
#             raise ValueError("Regressor not fitted yet.")
#         params = self._fill_missing_params(params)
#         X = np.array([[params[name] for name in self.param_names]])
#         X_norm = self._normalize_X(X)

#         # Predict mean
#         val = float(self.model.predict(X_norm)[0])

#         # Bootstrap uncertainty: std of predictions across bootstrap models
#         if self.bootstrap_models:
#             preds = [m.predict(X_norm)[0] for m in self.bootstrap_models]
#             conf = float(np.std(preds))
#         else:
#             conf = float(np.std(self.residuals)) if self.residuals else 0.0

#         return val, conf

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as SKGP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from typing import List, Dict, Tuple

class GaussianProcessRegressor(BaseRegressor):
    def __init__(self, alpha=1e-4, length_scale=None, length_scale_bounds=(0.05, 0.5), optimizer=None):
        """
        alpha: noise level (small positive value prevents exact interpolation)
        length_scale: float or list per dimension; default None sets 0.1 per dim
        length_scale_bounds: bounds for optimization if optimizer is used
        optimizer: 'fmin_l_bfgs_b' or None to disable hyperparameter optimization
        """
        super().__init__()
        self.alpha = alpha
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.optimizer = optimizer
        self.model = None
        self._X_min = None
        self._X_range = None

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        # Linear normalization to [0,1] based on min/max observed in fit
        return (X - self._X_min) / (self._X_range + 1e-12)

    def fit(self, samples: List[Dict]) -> Tuple[List[float], Dict]:
        self._compute_param_stats(samples)
        X = np.array([[s.parameters[name] for name in self.param_names] for s in samples])
        y = np.array([s.coefficient for s in samples])

        # Compute min/range for normalization
        self._X_min = X.min(axis=0)
        self._X_range = X.max(axis=0) - self._X_min

        X_norm = self._normalize_X(X)

        # Determine length scales
        n_features = X.shape[1]
        if self.length_scale is None:
            length_scale = [0.1] * n_features
        elif isinstance(self.length_scale, (float, int)):
            length_scale = [self.length_scale] * n_features
        else:
            length_scale = list(self.length_scale)

        kernel = C(1.0) * RBF(length_scale=length_scale, length_scale_bounds=[self.length_scale_bounds]*n_features)
        self.model = SKGP(kernel=kernel, alpha=self.alpha, optimizer=self.optimizer, normalize_y=True)
        self.model.fit(X_norm, y)

        y_pred, y_std = self.model.predict(X_norm, return_std=True)
        self.residuals = (y - y_pred).tolist()
        self.goodness_of_fit = {
            "rmse": float(np.sqrt(np.mean((y - y_pred) ** 2))),
            "n_samples": len(samples),
        }
        return self.residuals, self.goodness_of_fit

    def update(self, samples: List[Dict]) -> Tuple[List[float], Dict]:
        # sklearn GPs don't support incremental updates; refit
        return self.fit(samples)

    def __call__(self, **params) -> Tuple[float, float]:
        if self.model is None:
            raise ValueError("Regressor not fitted yet.")
        params = self._fill_missing_params(params)
        X = np.array([[params[name] for name in self.param_names]])
        X_norm = (X - self._X_min) / (self._X_range + 1e-12)
        val, std = self.model.predict(X_norm, return_std=True)
        return float(val[0]), float(std[0])



def selectRegressor(samples: List[Dict]) -> BaseRegressor: # TODO:  make this smarter
    if len(samples) > 10000:
        return KNearestNeighborRegressor()
    else:
        return GaussianProcessRegressor() #PolynomialRegressor(degree=3)