import numpy as np
from typing import List, Tuple, Dict, Union

from pyrse.flight_data import FlightData

class FlightDataAligner:
    def __init__(self, max_lag: float = 10.0):
        """
        Parameters
        ----------
        max_lag : float
            Maximum lag to search in seconds.
        """
        self.max_lag = max_lag

    def _get_weights(self, series, measured_w: float, calculated_w: float) -> np.ndarray:
        """Return weights array based on series source and interpolation mask."""
        n = len(series.values)
        if series.source in ['Calculated', 'Unknown']:
            return np.full(n, calculated_w, dtype=float)
        else:
            weights = np.full(n, measured_w, dtype=float)
            if hasattr(series, "interpolation_mask"):
                weights[series.interpolation_mask] = calculated_w
            return weights

    def _prep_series(self, fd, measured_w: float, calculated_w: float):
        """Extract series and corresponding weights."""
        series_dict = {}
        weights_dict = {}
        for key in ['h', 'Vz', 'azs']:
            s = fd[key]
            series_dict[key] = s
            weights_dict[key] = self._get_weights(s, measured_w, calculated_w)
        return series_dict, weights_dict

    def _weighted_cross_corr(self, ref_vals, target_vals, ref_w, target_w, max_lag_samples):
        """Weighted cross-correlation for all lags ±max_lag_samples."""
        n = len(ref_vals)
        lags = np.arange(-max_lag_samples, max_lag_samples + 1)
        corrs = []

        for lag in lags:
            if lag < 0:
                xs, ys = ref_vals[:lag], target_vals[-lag:]
                ws = ref_w[:lag] * target_w[-lag:]
            elif lag > 0:
                xs, ys = ref_vals[lag:], target_vals[:-lag]
                ws = ref_w[lag:] * target_w[:-lag]
            else:
                xs, ys = ref_vals, target_vals
                ws = ref_w * target_w

            if len(xs) == 0:
                corrs.append(-np.inf)
            else:
                num = np.sum(ws * xs * ys)
                den = np.sqrt(np.sum(ws * xs**2) * np.sum(ws * ys**2))
                corrs.append(num / den if den > 0 else -np.inf)

        return lags, np.array(corrs)

    def __call__(self, fds: List[Union["FlightData", Tuple["FlightData", Dict]]]):
        """
        Align a list of FlightData objects using derivative-aware interpolation.

        Returns
        -------
        dict : FlightData -> lag (seconds)
        int : index of reference FlightData
        """
        # Normalize inputs
        items = []
        for entry in fds:
            if isinstance(entry, tuple):
                fd, wdict = entry
                measured_w = wdict.get("MeasuredWeight", wdict.get("CalculatedWeight", 0.5)*2)
                calculated_w = wdict.get("CalculatedWeight", measured_w / 2)
            else:
                fd = entry
                measured_w, calculated_w = 1.0, 0.5
            items.append((fd, measured_w, calculated_w))

        # Choose coarsest FlightData as initial reference
        coarsest_idx = np.argmax([np.mean(np.diff(fd['h'].times)) for fd, _, _ in items])
        ref_fd, ref_mw, ref_cw = items[coarsest_idx]
        ref_series_dict, ref_weights_dict = self._prep_series(ref_fd, ref_mw, ref_cw)
        ref_dt = np.mean(np.diff(ref_series_dict['h'].times))
        max_lag_samples = int(round(self.max_lag / ref_dt))

        # Resample all flights to reference times using at(t)
        t_ref = ref_series_dict['h'].times
        interp_data = []
        for fd, mw, cw in items:
            series_dict, weights_dict = self._prep_series(fd, mw, cw)
            interp_series = {}
            interp_weights = {}
            for key in ['h', 'Vz', 'azs']:
                s = series_dict[key]
                interp_series[key] = np.array([s.at(t) for t in t_ref])
                interp_weights[key] = np.interp(t_ref, s.times, weights_dict[key], left=0.0, right=0.0)
            interp_data.append((fd, interp_series, interp_weights))

        # Preliminary lags relative to coarsest flight
        preliminary_lags = []
        for fd, series, weights in interp_data:
            corrs_total = None
            for key in ['h', 'Vz', 'azs']:
                lags, corrs = self._weighted_cross_corr(
                    ref_series_dict[key].values, series[key],
                    ref_weights_dict[key], weights[key], max_lag_samples
                )
                if corrs_total is None:
                    corrs_total = corrs
                else:
                    corrs_total += corrs
            best_lag = lags[np.argmax(corrs_total)]
            preliminary_lags.append(best_lag)

        preliminary_lags_sec = np.array(preliminary_lags) * ref_dt

        # Choose final reference as flight closest to center
        center_idx = np.argmin(np.abs(preliminary_lags_sec - np.median(preliminary_lags_sec)))
        final_ref_fd, final_ref_mw, final_ref_cw = items[center_idx]
        final_ref_series_dict, final_ref_weights_dict = self._prep_series(final_ref_fd, final_ref_mw, final_ref_cw)

        # Compute final lags relative to final reference
        final_lags_sec = {}
        for fd, series, weights in interp_data:
            corrs_total = None
            for key in ['h', 'Vz', 'azs']:
                lags, corrs = self._weighted_cross_corr(
                    np.array([final_ref_series_dict[key].at(t) for t in t_ref]),
                    series[key],
                    final_ref_weights_dict[key],
                    weights[key],
                    max_lag_samples
                )
                if corrs_total is None:
                    corrs_total = corrs
                else:
                    corrs_total += corrs
            best_lag = lags[np.argmax(corrs_total)]
            final_lags_sec[fd] = best_lag * ref_dt

        return final_lags_sec, center_idx
