import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUOrientationEstimator:
    def __init__(self, g=9.80665):
        """
        Parameters
        ----------
        g : float
            Reference gravitational acceleration (m/s^2).
        """
        self.g = g
        self.R_b2r = None
        self.R_r2i = None

    def _normalize(self, v):
        """Return a normalized version of the vector v."""
        v = np.asarray(v)
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            raise ValueError("Attempt to normalize near-zero vector")
        return v / norm

    def _average_vector(self, xs, ys, zs, mask):
        """Compute average vector over masked samples."""
        vx = np.mean(xs[mask])
        vy = np.mean(ys[mask])
        vz = np.mean(zs[mask])
        return np.array([vx, vy, vz])

    def _find_liftoff_index(self, axs, ays, azs, threshold):
        """Return the index of the first sample where acceleration magnitude exceeds threshold."""
        mags = np.linalg.norm(np.vstack([axs, ays, azs]), axis=0)
        above = np.where(mags > threshold)[0]
        if len(above) == 0:
            raise ValueError("No liftoff detected (threshold too high?)")
        return above[0]

    def _window_mask(self, ts, i0, window_time):
        """Return a mask selecting samples in [ts[i0], ts[i0]+window_time)."""
        t_start = ts[i0]
        t_end = t_start + window_time
        return (ts >= t_start) & (ts < t_end)

    def estimate(self, ts, axs_b, ays_b, azs_b, threshold=15.0, window_time=0.1):
        """
        Estimate IMU orientation using prelaunch gravity and first-motion vectors.

        Parameters
        ----------
        ts : array_like
            Time samples (seconds).
        axs_b, ays_b, azs_b : array_like
            Accelerations in body frame (m/s^2).
        threshold : float
            Magnitude threshold to detect liftoff (m/s^2).
        window_time : float
            Duration of averaging window after liftoff (seconds).

        Returns
        -------
        R_b2r : scipy.spatial.transform.Rotation
            Rotation from body frame to rocket frame (z-axis aligned with thrust axis).
        R_r2i : scipy.spatial.transform.Rotation
            Rotation from rocket frame to inertial frame (z-axis aligned with gravity).
        """
        ts = np.asarray(ts)
        axs_b, ays_b, azs_b = map(np.asarray, (axs_b, ays_b, azs_b))

        # Find liftoff index
        i0 = self._find_liftoff_index(axs_b, ays_b, azs_b, threshold)

        # Prelaunch gravity vector
        pre_mask = ts < ts[i0]
        g_b = self._average_vector(axs_b, ays_b, azs_b, pre_mask)
        g_b = self._normalize(g_b)

        # First motion vector
        post_mask = self._window_mask(ts, i0, window_time)
        f_b = self._average_vector(axs_b, ays_b, azs_b, post_mask)
        f_b = self._normalize(f_b)

        # Define rocket frame axes
        z_r = f_b                           # thrust axis
        x_r = self._normalize(np.cross(g_b, z_r))  # perpendicular to plane
        y_r = self._normalize(np.cross(z_r, x_r))  # completes right-handed basis

        # Rotation from body to rocket
        self.R_b2r = R.from_matrix(np.column_stack([x_r, y_r, z_r]).T)

        # Rocket-to-inertial: align g_b with [0,0,-1]
        g_r = self.R_b2r.apply(g_b)
        self.R_r2i = R.align_vectors([[0, 0, 1]], [g_r])[0] # TODO: FIGURE OUT IF THIS IS CORRECT. A GRAVITY VECTOR OF [0, 0, -1] WOULD MAKE MORE SENSE, BUT THIS WORKS.

        return self.R_b2r, self.R_r2i

    def _get_rotation(self, mode):
        if self.R_b2r is None or self.R_r2i is None:
            raise RuntimeError("Call estimate() before rotating data")
        if mode == "rocket":
            return self.R_b2r
        elif mode == "earth":
            return self.R_r2i * self.R_b2r
        else:
            raise ValueError("mode must be 'rocket' or 'earth'")

    def rotate_accels(self, axs, ays, azs, mode="earth"):
        """
        Rotate accelerometer arrays into rocket or earth frame.

        Parameters
        ----------
        axs, ays, azs : array_like
            Accelerometer measurements in body frame.
        mode : {'rocket', 'earth'}
            Frame to rotate into.

        Returns
        -------
        axs_r, ays_r, azs_r : ndarray
            Rotated accelerometer measurements.
        """
        rotation = self._get_rotation(mode)
        vecs = np.vstack([axs, ays, azs]).T
        rotated = rotation.apply(vecs)
        return rotated[:, 0], rotated[:, 1], rotated[:, 2]

    def rotate_gyros(self, gxs, gys, gzs, mode="earth"):
        """
        Rotate gyroscope arrays into rocket or earth frame.

        Parameters
        ----------
        gxs, gys, gzs : array_like
            Gyroscope measurements in body frame.
        mode : {'rocket', 'earth'}
            Frame to rotate into.

        Returns
        -------
        gxs_r, gys_r, gzs_r : ndarray
            Rotated gyroscope measurements.
        """
        rotation = self._get_rotation(mode)
        vecs = np.vstack([gxs, gys, gzs]).T
        rotated = rotation.apply(vecs)
        return rotated[:, 0], rotated[:, 1], rotated[:, 2]
