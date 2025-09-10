import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyrse.analysis import coefficients
from scipy.stats import qmc


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
    num_samples = 1000
    cd_noise_stddev = 0.1
    mach_range = (0.0, 0.5)
    aoa_deviation = 10.0

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

    # fig_2d, ax2d = plt.subplots(layout='constrained')
    # ax2d.set_xlabel('Mach')
    # ax2d.set_ylabel('Cd')
    # ax2d.set_title('Cd(Mach, AoA=0) with Asymmetric Transonic Peak')
    # ax2d.grid(True)
    # ax2d.set_xlim(0, 1.5)
    # ax2d.set_ylim(0, 2.5)
    # ax2d.scatter([s.parameters['M'] for s in samples],
    #           [s.coefficient for s in samples])
    # ax2d.scatter([s.parameters['M'] for s in noisy_samples],
    #           [s.coefficient for s in noisy_samples])
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, layout='constrained')
    ax.set_xlabel('Mach')
    ax.set_ylabel('AoA (deg)')
    ax.set_zlabel('Cd')
    ax.set_title('Cd(Mach, AoA) with Asymmetric Transonic Peak')
    ax.view_init(elev=30, azim=-120)
    ax.grid(True)
    ax.set_box_aspect([1,1,0.7])
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, 2.5)
    ax.scatter(
        [s.parameters['M'] for s in samples],
        [s.parameters['aoa'] for s in samples],
        [s.coefficient for s in samples],
        c='b', alpha=0.25
    )

    ax.scatter(
        [s.parameters['M'] for s in noisy_samples],
        [s.parameters['aoa'] for s in noisy_samples],
        [s.coefficient for s in noisy_samples],
        c='r', alpha=0.25
    )    
    plt.show()

