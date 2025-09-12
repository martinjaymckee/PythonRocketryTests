import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyrse.analysis import coefficients
from scipy.stats import qmc

import pyrse.analysis.coefficients as coeff # to avoid name conflict

def sigmoid(x):
    """Smooth transition function for blending regions."""
    return 0.5 * (1.0 + np.tanh(x / 2.0))

def cd_from_mach_aoa_asym_smooth(mach, aoa_deg, *, 
                                 Cd0=0.05,            
                                 k_aoa=0.02,          
                                 plateau_mult=0.85,   
                                 plateau_mach=0.6,    
                                 transonic_start=0.8,
                                 transonic_end=1.2,
                                 peak_mult=1.6,       
                                 sigma_left=0.05,     
                                 tau_right=0.25,      
                                 base_post_mult=0.6,  
                                 post_decay=0.5       
                                ):
    """
    Continuous Cd(Mach, AoA) model with:
      - exponential decay to plateau in subsonic,
      - asymmetric transonic peak (Gaussian left, exponential right),
      - smooth exponential supersonic decay toward asymptote.
    """

    # Ensure arrays (even for scalars)
    m = np.atleast_1d(np.asarray(mach, dtype=float))
    aoa = np.atleast_1d(np.asarray(aoa_deg, dtype=float))
    m, aoa = np.broadcast_arrays(m, aoa)

    # --- AoA scaling ---
    aoa_factor = 1.0 + k_aoa * (aoa**2)

    # --- Early exponential decrease ---
    k_decay = -np.log(max(1e-12, plateau_mult)) / max(1e-12, plateau_mach)
    early = np.exp(-k_decay * m)

    plateau = np.ones_like(m) * plateau_mult

    # --- Asymmetric transonic bump around Mach=1 ---
    A_peak = max(0.0, peak_mult - plateau_mult)
    bump = np.zeros_like(m)
    mask_left = (m <= 1.0)
    mask_right = ~mask_left
    if np.any(mask_left):
        bump[mask_left] = A_peak * np.exp(-0.5 * ((m[mask_left] - 1.0) / sigma_left)**2)
    if np.any(mask_right):
        bump[mask_right] = A_peak * np.exp(- (m[mask_right] - 1.0) / tau_right)
    transonic = plateau + bump

    # --- Continuous post-transonic decay ---
    if transonic_end <= 1.0:
        f_trans_end = plateau_mult + A_peak * np.exp(-0.5 * ((transonic_end - 1.0) / sigma_left)**2)
    else:
        f_trans_end = plateau_mult + A_peak * np.exp(- (transonic_end - 1.0) / tau_right)

    post = base_post_mult + (f_trans_end - base_post_mult) * np.exp(-post_decay * (m - transonic_end))
    post = np.where(m <= transonic_end, f_trans_end, post)  # safe scalar/array handling

    # --- Smooth blending between regimes ---
    d_early = 0.06 + 0.02 * plateau_mach
    d_plateau_lo = 0.08
    d_trans = 0.06
    d_post = 0.08

    w_early = sigmoid((plateau_mach - m) / d_early)
    w_plateau = sigmoid((m - plateau_mach) / d_plateau_lo) * sigmoid((transonic_start - m) / d_trans)
    w_trans = sigmoid((m - transonic_start) / d_trans) * sigmoid((transonic_end - m) / d_trans)
    w_post = sigmoid((m - transonic_end) / d_post)

    weight_sum = w_early + w_plateau + w_trans + w_post + 1e-12
    mult = (w_early * early + w_plateau * plateau + w_trans * transonic + w_post * post) / weight_sum

    Cd = Cd0 * aoa_factor * mult

    # Return scalar if input was scalar
    if np.isscalar(mach) and np.isscalar(aoa_deg):
        return float(Cd)
    return Cd



if __name__ == '__main__':
    seed = 42
    num_samples = 5000
    cd_noise_stddev = 0.1
    mach_range = (0.0, 1.5)
    mach_range_display = (0.0, 1.6)
    aoa_deviation = 8.0
    aoa_deviation_display = 10.0

    # Cd model parameters
    cd_params = {
        "Cd0": 0.85,
        "k_aoa": .0035,  # AoA quadratic coefficient (deg^-2)
        "plateau_mult": 0.59,  # 0.5 / 0.65 -> Cd at M=0.3 target
        "plateau_mach": 0.3,
        "transonic_start": 0.8,
        "transonic_end": 1.2,
        "peak_mult": 1.927465,   # tuned so Cd(M=1, AoA=0) ≈ 1.2
        "sigma_left": 0.04,      # Gaussian width on subsonic side
        "tau_right": 0.28,       # exponential decay scale on supersonic side
        "base_post_mult": 0.6,
        "post_decay": 0.45
    }

    def lhs_samples(n_samples, mach_range=(0.0, 3.0), aoa_range=(-10, 10), seed=None):
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        unit_samples = sampler.random(n=n_samples)  # shape (n_samples, 2)

        # Scale to physical ranges
        scaled = qmc.scale(unit_samples,
                        l_bounds=[mach_range[0], aoa_range[0]],
                        u_bounds=[mach_range[1], aoa_range[1]])

        mach = scaled[:, 0]
        aoa  = scaled[:, 1]
        return mach, aoa

    def create_cd_samples(mach_range, aoa_range, num_samples, seed=None):
        """Generate a grid of Cd samples over specified Mach and AoA ranges."""
        machs, aoas = lhs_samples(num_samples, mach_range, aoa_range, seed=seed)

        cds = cd_from_mach_aoa_asym_smooth(machs, aoas, **cd_params)
        samples = [coefficients.CoefficientSample(
                    coefficient=cd,
                    parameters={'M': m, 'aoa': aoa},
                    weight=1.0
                ) for cd, m, aoa in zip(cds, machs, aoas)]
        return samples

    def add_gaussian_noise_to_samples(samples, stddev=0.01, seed=None):
        """
        Returns a new list of CoefficientSamples with Gaussian noise added to the coefficient value.
        """
        rng = np.random.default_rng(seed)
        noisy_samples = []
        for s in samples:
            noisy_coeff = s.coefficient + rng.normal(0, stddev)
            noisy_samples.append(coefficients.CoefficientSample(
                coefficient=noisy_coeff,
                parameters=s.parameters,
                weight=s.weight
            ))
        return noisy_samples
    
    # Generate samples
    samples = create_cd_samples(mach_range, (-aoa_deviation, aoa_deviation), num_samples, seed=seed)
    noisy_samples = add_gaussian_noise_to_samples(samples, stddev=cd_noise_stddev, seed=seed)

    cd_mapper = coeff.CoefficientMapping(noisy_samples)
    
    ms = np.array([s.parameters['M'] for s in samples]) 
    aoas = np.array([s.parameters['aoa'] for s in samples])
    cds_clean = np.array([s.coefficient for s in samples])
    cds_noisy = np.array([s.coefficient for s in noisy_samples])

    cds_mapped = np.array([cd_mapper({'M': m, 'aoa': aoa})[0] for m, aoa in zip(ms, aoas)])

    fig_2d, ax2d = plt.subplots(layout='constrained')
    ax2d.set_xlabel('Mach')
    ax2d.set_ylabel('Cd')
    ax2d.set_title('Cd(Mach, AoA=0) with Asymmetric Transonic Peak')
    ax2d.grid(True)
    ax2d.set_xlim(*mach_range_display)
    ax2d.set_ylim(-aoa_deviation_display, aoa_deviation_display)
    ax2d.scatter(ms, aoas, c=cds_clean, cmap='viridis', s=30, alpha=0.7)

    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, layout='constrained')
    ax.set_xlabel('Mach')
    ax.set_ylabel('AoA (deg)')
    ax.set_zlabel('Cd')
    ax.set_title('Cd(Mach, AoA) with Asymmetric Transonic Peak')
    ax.view_init(elev=30, azim=-120)
    ax.grid(True)
    ax.set_box_aspect([1,1,0.7])
    ax.set_xlim(*mach_range_display)
    ax.set_ylim(-aoa_deviation_display, aoa_deviation_display)
    ax.set_zlim(np.min(cds_noisy)-0.1, np.max(cds_noisy)+0.1)
    ax.scatter(ms, aoas, cds_clean, c='b', s=20, alpha=0.7)
    ax.scatter(ms, aoas, cds_noisy, c='r', s=20, alpha=0.2)
    ax.scatter(ms, aoas, cds_mapped, c='g', s=20, alpha=0.7)

    fig_residuals, ax_residuals = plt.subplots(1, layout='constrained')
    ax_residuals.set_xlabel('sample index')
    ax_residuals.set_ylabel('Cd residual')
    ax_residuals.set_title('Cd Fit Residuals')
    ax_residuals.grid(True)
    residuals_noisy = cds_noisy - cds_clean
    residuals_mapped = cds_mapped - cds_clean
    ax_residuals.scatter(np.arange(len(residuals_noisy)), residuals_noisy, c='r', s=20, alpha=0.2, label='noisy samples')
    ax_residuals.scatter(np.arange(len(residuals_mapped)), residuals_mapped, c='g', s=20, alpha=0.7, label='mapped samples')
    ax_residuals.axhline(0, color='k', lw=0.8)
    ax_residuals.legend()  
    print('Noisy samples: rms(residual) = {:.4f}, stddev = {:.4f}'.format(np.sqrt(np.sum(residuals_noisy**2))/len(residuals_noisy), np.std(residuals_noisy)))
    print('Mapped samples: rms(residual) = {:.4f}, stddev = {:.4f}'.format(np.sqrt(np.sum(residuals_mapped**2))/len(residuals_mapped), np.std(residuals_mapped)))

    # import pyrse.analysis.regression as regres
    # for gamma in [0.5, 1.0, 2.0, 5.0, 10.0]:
    #     for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
    #         rbf = regres.RBFRegressor(alpha=alpha, gamma=gamma, n_bootstrap=10)
    #         res, gof = rbf.fit(samples)
    #         print(f"gamma={gamma}, alpha={alpha}, rmse={gof['rmse']:.4f}")

    r1 = np.abs(residuals_noisy)
    r2 = np.abs(residuals_mapped)
    vmin = min(r1.min(), r2.min())
    vmax = max(r1.max(), r2.max())

    fig_noisy_2d, ax_noisy_2d = plt.subplots(layout='constrained')
    ax_noisy_2d.set_xlabel('Mach')
    ax_noisy_2d.set_ylabel('Aoa (deg)')
    ax_noisy_2d.set_title('Cd Error Magnitude in Noisy Samples')
    ax_noisy_2d.grid(True)
    ax_noisy_2d.set_xlim(*mach_range_display)
    ax_noisy_2d.set_ylim(-aoa_deviation_display, aoa_deviation_display)
    ax_noisy_2d.scatter(ms, aoas, c=r1, cmap='viridis', s=30, alpha=0.7, vmin=vmin, vmax=vmax)

    fig_mapped_2d, ax_mapped_2d = plt.subplots(layout='constrained')
    ax_mapped_2d.set_xlabel('Mach')
    ax_mapped_2d.set_ylabel('Aoa (deg)')
    ax_mapped_2d.set_title('Cd Error Magnitude in Mapped Samples')
    ax_mapped_2d.grid(True)
    ax_mapped_2d.set_xlim(*mach_range_display)
    ax_mapped_2d.set_ylim(-aoa_deviation_display, aoa_deviation_display)
    ax_mapped_2d.scatter(ms, aoas, c=r2, cmap='viridis', s=30, alpha=0.7, vmin=vmin, vmax=vmax)


    plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from numpy.linalg import cond, eigvals

# # X, y, weights must be from your fit call
# # If you don't have X currently stored, temporarily store it in fit() as self._last_X, self._last_y, self._last_w

# X = reg._last_X          # (n, p)
# y = reg._last_y          # (n,)
# w = reg._last_w          # (n,)

# # 1) Condition number of weighted normal matrix
# W_sqrt = np.sqrt(w)
# Xw = X * W_sqrt[:, None]
# XTWX = Xw.T @ Xw
# print("cond(X^T W X) =", cond(XTWX))

# # 2) Small eigenvalues?
# vals = np.sort(np.abs(eigvals(XTWX)))[::-1]
# print("Top 6 eigenvalues:", vals[:6])
# print("Last 6 eigenvalues:", vals[-6:])

# # 3) Leverage (hat matrix diagonal) -- indicates influential points
# # H = W_sqrt[:,None]*X @ inv(XTWX) @ (W_sqrt[:,None]*X).T  but only need diag:
# XTWX_inv = np.linalg.pinv(XTWX)
# H_diag = np.sum((Xw @ XTWX_inv) * Xw, axis=1)   # elementwise dot
# plt.figure()
# plt.scatter(np.arange(len(H_diag)), H_diag, s=10)
# plt.title("Leverage (hat diag) per sample")
# plt.show()
# print("High leverage indices:", np.where(H_diag > 3*X.shape[1]/X.shape[0])[0])

# # 4) Residuals vs predicted and vs inputs
# y_pred = X @ reg.coeffs
# residuals = y - y_pred
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.scatter(y_pred, residuals, s=5); plt.axhline(0, color='k', lw=0.8)
# plt.title("Residuals vs Predicted")
# plt.subplot(1,2,2)
# plt.hist(residuals, bins=60); plt.title("Residual histogram")
# plt.show()

# # 5) Residuals vs each parameter (check structured misfit)
# for i,name in enumerate(reg.param_names):
#     plt.figure()
#     plt.scatter([s.parameters[name] for s in reg._last_samples], residuals, s=6)
#     plt.xlabel(name); plt.ylabel("residual"); plt.title(f"Residual vs {name}")
#     plt.show()

# def fit_weighted_ridge(X, y, w, alpha=1e-6):
#     # X: (n,p), w: (n,)
#     W_sqrt = np.sqrt(w)
#     Xw = X * W_sqrt[:, None]
#     yw = y * W_sqrt
#     XT_W_X = Xw.T @ Xw
#     # Add ridge: alpha * I (alpha chosen by CV)
#     p = XT_W_X.shape[0]
#     A = XT_W_X + alpha * np.eye(p)
#     b = Xw.T @ yw
#     coeffs = np.linalg.solve(A, b)
#     # residuals (unweighted)
#     y_pred = X @ coeffs
#     residuals = y - y_pred
#     sigma2 = np.sum(w * residuals**2) / (len(y) - p)
#     cov = sigma2 * np.linalg.inv(A)
#     return coeffs, cov, residuals


# Using PolynomialFeatures with raw inputs can produce correlated columns. Use StandardScaler first, or use sklearn.preprocessing pipeline and orthonormal polynomials:
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.pipeline import Pipeline

# pipeline = Pipeline([
#     ('scale', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, include_bias=True))
# ])
# X_scaled_poly = pipeline.fit_transform(raw_X)   # store pipeline in reg
# Scaling massively improves conditioning and interpretability; do this before building the X you store in diagnostics.

# import numpy as np
# from numpy.linalg import solve, pinv
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.pipeline import Pipeline

# class StablePolyRegressor:
#     def __init__(self, degree=2, ridge_alpha=1e-6, scale_inputs=True):
#         self.degree = degree
#         self.ridge_alpha = ridge_alpha
#         self.scale_inputs = scale_inputs
#         steps = []
#         if scale_inputs:
#             steps.append(('scale', StandardScaler()))
#         steps.append(('poly', PolynomialFeatures(degree=degree, include_bias=True)))
#         self.pipeline = Pipeline(steps)
#         self.coeffs = None
#         self.cov = None
#         self.p = None
#         # store last batch for diagnostics
#         self._last_X = None
#         self._last_y = None
#         self._last_w = None

#     def fit(self, raw_X, y, weights=None):
#         X = self.pipeline.fit_transform(raw_X)
#         self._last_X = X; self._last_y = y; self._last_w = weights if weights is not None else np.ones(len(y))
#         if weights is None:
#             weights = np.ones(len(y))
#         W_sqrt = np.sqrt(weights)
#         Xw = X * W_sqrt[:, None]
#         yw = y * W_sqrt
#         XT_W_X = Xw.T @ Xw
#         p = XT_W_X.shape[0]
#         A = XT_W_X + self.ridge_alpha * np.eye(p)
#         b = Xw.T @ yw
#         self.coeffs = solve(A, b)
#         residuals = y - X @ self.coeffs
#         sigma2 = np.sum(weights * residuals**2) / (len(y) - p)
#         self.cov = sigma2 * pinv(A)
#         self.p = p
#         return residuals

#     def predict(self, raw_X):
#         X = self.pipeline.transform(raw_X)
#         mu = X @ self.coeffs
#         var = np.sum((X @ self.cov) * X, axis=1)  # row @ cov @ row^T for each
#         return mu, np.sqrt(np.maximum(var, 0.0))
