import math
import random

import matplotlib.pyplot as plt
import numpy as np

import adc_models
import normal_rvs as nrvs
import pid


def wireDiametermm(awg):
    d = .460 * pow(0.890625, (awg+3))
    return 25.4 * d


def wireResistance(L, awg, rho=None):
    rho = 7.4e-5/100 if rho is None else rho  # ohm-m
    d = wireDiametermm(awg)/1000  # m
    A = math.pi * pow(d/2, 2)
    return (rho * L) / A


def driverCharacteristics(L, awg, P, rho=None):
    R = wireResistance(L, awg, rho)
    V = math.sqrt(P*R)
    I = V / R
    return V, I


class WireMaterial:
    def __init__(self, density, conductivity, tcr, Cp, thermal_expansion, T0=293.15):
        self.__density = density
        self.__conductivity = conductivity
        self.__tcr = tcr
        self.__Cp = Cp
        self.__thermal_expansion = thermal_expansion
        self.__T0 = T0

    @property
    def density(self):
        return self.__density

    @property
    def conductivity(self):
        return self.__conductivity

    @property
    def tcr(self):
        return self.__tcr

    @property
    def Cp(self):
        return self.__Cp

    @property
    def thermal_expansion(self):
        return self.__thermal_expansion

    @property
    def T0(self):
        return self.__T0


class HotWire:
    __materials = {
        '316L': WireMaterial(8000, 7.4e-5, 0.00092, 500, 17.2e-6),
        'Nikrothal 60': WireMaterial(8000, 1.11e-4, 0.000111, 460, 18e-6),  # Emissivity = 0.88
        'Nikrothal 80': WireMaterial(8300, 1.09e-4, 5.83e-5, 460, 18e-6),  # Emissivity = 0.88
        'Ni200': WireMaterial(8900, 9e-6, 0.0061, 480, 13.3e-6),
        'Kanthal A1': WireMaterial(7100, 1.45e-4, 4e-5, 460, 14e-6),  # Emissivity = 0.7
        'Nifethal 70': WireMaterial(8450, 2.0e-5, .00532, 520, 15e-6),  # Emissivity = 0.88
        'Nifethal 52': WireMaterial(8200, 3.7e-5, .00354, 520, 10e-6)  # Emissivity = 0.88
    }

    def __init__(self, material_type, awg, L, T0=293.15+20):
        self.__material_name = material_type
        self.__material = HotWire.__materials[material_type]
        self.__L = L
        self.__awg = awg
        self.__r = wireDiametermm(self.__awg) / 2000

        self.__area = math.pi * (2 * self.__r) * self.length
        self.__crosssectional_area = math.pi * (self.__r**2)
        self.__volume = self.__crosssectional_area * self.length
        self.__mass = self.__material.density * self.__volume
        self.__R0 = (self.__material.conductivity * self.length) / self.crosssectional_area / 100
        self.__T0 = T0
        self.__T = T0
        self.__Estored = self.mass * self.__material.Cp * (self.__T - self.__T0)

    def __str__(self):
        return 'Wire(material = {}, awg = {}, L = {})'.format(self.__material_name, self.__awg, self.__L)

    @property
    def available_materials(self):
        return self.__materials.keys()

    @property
    def material(self):
        return self.__material

    @property
    def length(self):
        return self.__L

    @property
    def diameter(self):
        return 2*self.__r

    @property
    def area(self):
        return self.__area

    @property
    def crosssectional_area(self):
        return self.__crosssectional_area

    @property
    def volume(self):
        return self.__volume

    @property
    def mass(self):
        return self.__mass

    @property
    def R0(self):
        return self.__R0

    @property
    def R(self):
        return self.rFromT(self.T)

    @property
    def T0(self):
        return self.__T0

    @property
    def T(self):
        return self.__T

    @T.setter
    def T(self, v):
        self.__T = v
        return self.__T

    def tFromR(self, R):
        if R == self.R0:
            return self.T0
        return (((R / self.R0) - 1) / self.__material.tcr) + self.__material.T0

    def rFromT(self, T):
        if T == self.__material.T0:
            return self.R0
        return self.__R0 * (1 + self.__material.tcr * (T - self.__material.T0))

    def update(self, dt, Idrv, Pload=0, Tamb=None):
        data = {}
        Tamb = self.T0 if Tamb is None else Tamb
        hc = 12.12  #- 1.16 v + 11.6 v^2
        emissivity = 1.0
        k = 1.38064852e-23  # Stefan-Boltzmann constant
        Cp = self.__material.Cp
        A = self.area
        R = self.R
        data['R'] = R
        T = self.T
        m = self.mass
        data['I'] = Idrv
        Ein = (Idrv**2) * R * dt
        Econv = hc * A * (T - Tamb) * dt
        Erad = emissivity * A * k * ((T - Tamb)**4) * dt
        Estored = m * Cp * (T - Tamb)
        E = Estored + Ein - Econv - Erad - (Pload * dt)
        self.__T = (E / (m * Cp)) + Tamb
        data['T'] = self.__T
        return data


class ResistanceEstimator:
    def __init__(self):
        self.__R0 = None
        self.__R_last = None
        self.__R_diss = 1

    @property
    def R0(self):
        return self.__R0

    @R0.setter
    def R0(self, R):
        self.__R0 = R
        return self.__R0

    def __call__(self, V, I):
        R = None
        if nrvs.mean(I) > 3 * nrvs.standard_deviation(I):
            R = V / I
        else:
            R = self.__R_last
            if R is not None:
                R *= self.__R_diss
        if R is None:
            R = self.__R0
        self.__R_last = nrvs.NRV.Construct(R)
        return self.__R_last


class TemperatureEstimator:
    def __init__(self, wire):
        self.__R0 = None
        self.__T0 = None
        self.__wire = wire

    @property
    def R0(self):
        return self.__R0

    @R0.setter
    def R0(self, R):
        self.__R0 = R
        return self.__R0

    @property
    def T0(self):
        return self.__T0

    @T0.setter
    def T0(self, T):
        self.__T0 = T
        return self.__T0

    @property
    def tcr(self):
        return self.__wire.material.tcr

    def __call__(self, R, Tamb):
        Test = self.T0 if R == self.R0 else (((R / self.R0) - 1) / self.tcr) + self.T0
        return max(Tamb, Test)
    #
    # def tFromR(self, R, Tamb):
    #     if R == self.R0:
    #         return self.T0
    #     return (((R / self.R0) - 1) / self.__material.tcr) + self.__material.T0
    #
    # def rFromT(self, T):
    #     if T == self.__material.T0:
    #         return self.R0
    #     return self.__R0 * (1 + self.__material.tcr * (T - self.__material.T0))


class FeedbackFilter:
    def __init__(self, dt):
        self.__dt = dt

    def __str__(self):
        return 'FeedbackFilter()'

    @property
    def dt(self):
        return self.__dt

    @dt.setter
    def dt(self, v):
        self.__dt = v
        self.doFilterCoefficientUpdate()
        return self.__dt

    def __call__(self, v, dt=None, **kwargs):
        if dt is not None:
            self.dt = dt
        return nrvs.NRV.Construct(v)

    def doFilterCoefficientUpdate(self):
        pass


class RecursiveFeedbackFilter(FeedbackFilter):
    def __init__(self, dt=0.01, e=0.75):
        FeedbackFilter.__init__(self, dt)
        self.__e = e
        self.__v = None

    def __str__(self):
        return 'RecursiveFeedbackFilter(e = {})'.format(self.e)

    @property
    def e(self):
        return self.__e

    @e.setter
    def e(self, v):
        self.__e = v
        return self.__e

    def __call__(self, v, dt=None, e=None, **kwargs):
        if dt is not None:
            self.dt = dt
        if e is not None:
            self.e = e
        if self.__v is None:
            self.__v = nrvs.NRV.Construct(v)
        else:
            self.__v = self.e*self.__v + (1-self.__e)*v
        return self.__v

    def doFilterCoefficientUpdate(self):
        pass


class AlphaBetaFeedbackFilter(FeedbackFilter):
    def __init__(self, dt, sigma_n=0.1, sigma_p=0.01):
        FeedbackFilter.__init__(self, dt)
        self.__alpha = 0
        self.__beta = 0
        self.__sigma_n = sigma_n
        self.__sigma_p = sigma_p
        self.__v = None
        self.__dv = None
        self.__s = None
        self.doFilterCoefficientUpdate()

    def __str__(self):
        return 'AlphaBetaFeedbackFilter(alpha = {}, beta = {}, sigma_p = {}, sigma_n = {}, dt = {})'.format(self.alpha, self.beta, self.sigma_p, self.sigma_n, self.dt)

    @property
    def sigma_n(self):
        return self.__sigma_n

    @sigma_n.setter
    def sigma_n(self, v):
        self.__sigma_n = v
        self.doFilterCoefficientUpdate()
        return self.__sigma_n

    @property
    def sigma_p(self):
        return self.__sigma_p

    @sigma_p.setter
    def sigma_p(self, v):
        self.__sigma_p = v
        self.doFilterCoefficientUpdate()
        return self.__sigma_p

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    def __call__(self, v, dt=None, sigma_n=None, **kwargs):
        if dt is not None:
            self.dt = dt
        if nrvs.is_nrv(v):
            self.sigma_n = v.standard_deviation
        else:
            if sigma_n is not None:
                self.sigma_n = sigma_n
        if self.__v is None:
            self.__v = nrvs.mean(v)
            self.__dv = 0
        # Temperature Prediction
        v_pre = self.__v + (self.dt * self.__dv)
        # Temperature Residual
        r = nrvs.mean(v) - v_pre
        # Temperature Correction
        self.__v = v_pre + (self.__alpha * r)
        self.__dv += ((self.__beta / self.dt) * r)
        return nrvs.NRV(self.__v, variance=self.__s)

    def doFilterCoefficientUpdate(self):
        if self.sigma_n == 0:
            self.__alpha = 0.8
            self.__beta = 0.005
        else:
            lam = (self.sigma_p * self.dt) / self.sigma_n
            r = (4 + lam - math.sqrt(8*lam + lam**2)) / 4
            self.__alpha = 1 - r**2
            self.__beta = (2 * (2 - self.alpha)) - (4 * math.sqrt(1 - self.alpha))
        # Innovation Variance
        self.__s = (self.sigma_n**2) / (1 - (self.alpha**2))


class PredictiveTemperatureFilter(FeedbackFilter):
    def __init__(self, dt, wire=None, Tamb=None):
        FeedbackFilter.__init__(self, dt)
        self.__wire = wire
        self.__R_est = ResistanceEstimator()
        self.__T_last = Tamb
        self.__R_last = self.__wire.rFromT(Tamb)
        self.doFilterCoefficientUpdate()

    def __str__(self):
        return 'PredictiveTemperatureFilter(wire = {})'.format(self.__wire)

    def __call__(self, T_meas, R_meas=0, I_drv=0, **kwargs):
        hc = 12.12  #- 1.16 v + 11.6 v^2
        emissivity = 1.0
        k = 1.38064852e-23  # Stefan-Boltzmann constant

        T_meas = nrvs.NRV.Construct(T_meas)
        T0 = self.__wire.material.T0
        # R_meas = self.__wire.rFromT(self.__T_last)
        C0 = I_drv**2 * R_meas * self.dt
        C1 = hc * self.__wire.area * self.dt
        C2 = emissivity * self.__wire.area * k * self.dt
        C3 = self.__wire.mass * self.__wire.material.Cp
        Tn = ((C0 - (C1 * (self.__T_last - T0)) - (C2 * ((self.__T_last - T0)**4))) / C3) + self.__T_last
        Tn += nrvs.NRV.Noise(sd=0.05)
        var_T_meas = nrvs.variance(T_meas)
        var_Tn = nrvs.variance(Tn)
        var_sum = var_T_meas + var_Tn
        K0 = 0 if var_sum == 0 else var_T_meas / var_sum
        K1 = abs((nrvs.mean(Tn) - nrvs.mean(T_meas)) / max(nrvs.mean(Tn), nrvs.mean(T_meas)))
        # K = 0.75
        K = K0
        self.__T_last = (K * Tn) + ((1 - K) * T_meas)
        # print('T_meas = {}, Tn = {}, K = {} -> T = {}'.format(T_meas, Tn, K, self.__T_last))
        return self.__T_last

    def doFilterCoefficientUpdate(self):
        pass


class HotwireController:
    def __init__(self, Vin, wire, Tamb, filt_T=None, filt_R=None, Vmax=28, Tset=400, Imax=10):
        self.__Vin = Vin
        self.__Tset = Tset
        self.__Tamb = Tamb
        self.__T = None
        self.__wire = wire
        self.__Vmax = Vmax
        self.__Vdrv = 0
        self.__Imax = Imax
        self.__R_est = ResistanceEstimator()
        self.__T_est = TemperatureEstimator(self.__wire)
        self.__filt_T = RecursiveFeedbackFilter() if filt_T is None else filt_T
        self.__filt_R = RecursiveFeedbackFilter() if filt_R is None else filt_R

    def __str__(self):
        fmt = '{}():\n\t\tTemperature Filter = {}\n\t\tResistance Filter = {}'
        return fmt.format(self.__class__.__name__, self.__filt_T, self.__filt_R)

    @property
    def wire(self):
        return self.__wire

    @property
    def R_est(self):
        return self.__R_est

    @property
    def T_est(self):
        return self.__T_est

    @property
    def Tset(self):
        return self.__Tset

    @Tset.setter
    def Tset(self, T):
        self.__Tset = T
        self.doTsetUpdate(T)
        return self.__Tset

    @property
    def Tamb(self):
        return self.__Tamb

    @Tamb.setter
    def Tamb(self, v):
        self.__Tamb = v
        return self.__Tamb

    @property
    def T(self):
        return self.__T

    @T.setter
    def T(self, v):
        self.__T = v
        return self.__T

    @property
    def Vmax(self):
        return self.__Vmax

    @property
    def Vin(self):
        return self.__Vin

    @property
    def Vdrv(self):
        return self.__Vdrv

    @Vdrv.setter
    def Vdrv(self, V):
        self.__Vdrv = V
        return self.__Vdrv

    def estimateTandR(self, V_hw, I_hw):
        R = self.__R_est(V_hw, I_hw)
        T = self.__T_est(R, self.Tamb)
        if T < self.Tamb:
            print('R = {}, T = {}'.format(R, T))
            T = self.Tamb
        return T, R

    def filter_estimates(self, dt, T_meas, R_meas, V_meas, I_meas):
        T_filt = self.__filt_T(T_meas, dt=dt, I_drv=I_meas, R_meas=R_meas)
        R_filt = self.__filt_R(R_meas, dt=dt, I_drv=I_meas)
        return T_filt, R_filt

    def update(self, dt, V_drv, V_hw, I_hw, Pload=0, Tset=None):
        assert False, 'Error: {}.update() is not implemented!'.format(self.__class__.__name__)


class PredictiveHotwireController(HotwireController):
    def __init__(self, Vin, wire, Tamb, filt_T=None, filt_R=None, Vmax=28, Tset=400, Imax=10, Kp=0.25, Ki=0.0025, Kd=0):
        HotwireController.__init__(self, Vin, wire, Tamb, filt_T, filt_R, Vmax, Tset, Imax)
        self.__correction_pid = pid.PID(Kp=Kp, Ki=Ki, Kd=Kd, limiter=pid.SaturationLimiter(-Vmax, Vmax), dpp_filter=pid.RecursiveSmoothingFilter(0.995))
        self.__correction_pid.sp = Tset

    @property
    def pid(self):
        return self.__correction_pid

    def update(self, dt, V_drv, V_hw, I_hw, Pload=0, Tset=None):
        data = {}
        if Tset is not None:
            self.Tset = Tset
        hc = 12.12  #- 1.16 v + 11.6 v^2
        emissivity = 1.0
        k = 1.38064852e-23  # Stefan-Boltzmann constant
        A = self.wire.area
        T_est, R_est = self.estimateTandR(V_hw, I_hw)
        self.T, R_est = self.filter_estimates(dt, T_est, R_est, V_hw, I_hw)
        data['T_est'] = self.T
        data['R_est'] = R_est
        Pconv = hc * A * (self.Tset - self.wire.T0)
        Prad = emissivity * A * k * ((self.Tset - self.wire.T0)**4)
        Ptotal = (Pconv + Prad + Pload)
        # print('Estimated T = {}, R = {}'.format(self.T, R_est))
        Vtgt = 0 if Ptotal <= 0 else R_est * math.sqrt(Ptotal / R_est)
        data['Vtgt'] = Vtgt
        Vcorrect, pid_data = self.__correction_pid(dt, self.T, debug=True)
        # print('Vcorrect = {}'.format(Vcorrect))
        data['Vcorrect'] = Vcorrect
        data.update(pid_data)
        self.Vdrv = max(0, min(self.Vmax, Vtgt + Vcorrect))
        # print('Vdrv = {}'.format(self.Vdrv))
        data['duty_cycle'] = self.Vdrv / self.Vin
        # print('duty_cycle = {}'.format(data['duty_cycle']))
        return self.Vdrv, data

    def doTsetUpdate(self, T):
        self.__correction_pid.sp = T


class PIDHotwireController(HotwireController):
    def __init__(self, Vin, wire, Tamb, filt_T=None, filt_R=None, Vmax=28, Tset=400, Imax=10, Kp=0.2, Ki=0.02, Kd=0):
        HotwireController.__init__(self, Vin, wire, Tamb, filt_T, filt_R, Vmax, Tset, Imax)
        self.__pid = pid.PID(Kp=Kp, Ki=Ki, Kd=Kd, limiter=pid.SaturationLimiter(-Vmax, Vmax), dpp_filter=pid.RecursiveSmoothingFilter(0.9999))
        self.__pid.sp = Tset

    @property
    def pid(self):
        return self.__pid

    def update(self, dt, V_drv, V_hw, I_hw, Pload=0, Tset=None):
        data = {}
        if Tset is not None:
            self.Tset = Tset
            self.__pid.sp = Tset
        T = self.estimateT(V_hw, I_hw)
        data['T_est'] = T
        Vtgt, pid_data = self.__pid(dt, T, debug=True)
        data['Vtgt'] = Vtgt
        data['Vcorrect'] = 0
        data.update(pid_data)
        self.Vdrv = max(0, min(self.Vmax, Vtgt))
        data['duty_cycle'] = self.Vdrv / self.Vin
        return self.Vdrv, data

    def doTsetUpdate(self, T):
        self.__pid.sp = Tset


class HotwireCutSimulation:
    def default_v_cut_func(t):
        if t < 3:
            return 0.
        elif t < 20:
            return 3e-3
        return 0.75e-3

    def __init__(self, w_cut=None, v_cut_func=None, Tamb=273.15 + 23):
        self.__w_cut = 1 if w_cut is None else w_cut
        self.__v_cut_func = HotwireCutSimulation.default_v_cut_func if v_cut_func is None else v_cut_func
        self.__foam_density = 24.8  # kg/m^3
        self.__Cp_foam = 1500  # J/kgK
        self.__T_melting_foam = 273.15 + 270  # K
        self.__Tamb = Tamb

    @property
    def w_cut(self):
        return self.__w_cut

    @w_cut.setter
    def w_cut(self, w):
        self.__w_cut = w
        return self.__w_cut

    def reset(self, Tinit=None):
        if Tinit is not None:
            self.__Tamb = Tinit

    def __call__(self, t, wire):
        v_cut = self.__v_cut_func(t)
        Pload = 0
        if v_cut > 0 and wire.T > self.__T_melting_foam:  # cutting only happens if the wire is above the melting point
            Vcut = v_cut * self.__w_cut * wire.diameter
            dT = self.__T_melting_foam - self.__Tamb
            Pload = self.__Cp_foam * self.__foam_density * dT * Vcut
        return Pload


class HotwireAFE:
    def __init__(self,  wire, samples=64, V_gain=0.09, I_gain=50, R_I=10e-3, R_w=1,
            Vref=2.5, BW=10e3, adc_class=adc_models.LPCChannel):
        self.__wire = wire
        self.__samples = samples
        self.__V_gain = V_gain
        self.__I_gain = I_gain
        self.__R_I = R_I
        self.__R_w = R_w
        self.__Vref = Vref
        self.__adc_I = adc_class(Vref)
        self.__I_amp_noise = nrvs.NRV.Noise(sd=75e-9 * math.sqrt(BW))  # INA186
        self.__adc_V = adc_class(Vref)
        self.__V_amp_noise = nrvs.NRV.Noise(sd=8e-9 * math.sqrt(BW))  # INA821
        self.__adc_V_drv = adc_class(Vref)
        self.__V_drv_gain = V_gain
        self.__V_drv_noise = nrvs.NRV.Noise(sd=50e-3)

    @property
    def ranges(self):
        def scaleFSR(adc, g):
            v_min, v_max = adc.Vfs
            return (g * v_min, g * v_max)

        return {
            'V_drv': scaleFSR(self.__adc_V_drv, 1 / self.__V_drv_gain),
            'I_drv': scaleFSR(self.__adc_I, 1 / (self.__R_I * self.__I_gain)),
            'V_hw': scaleFSR(self.__adc_V, 1 / self.__V_gain)
        }

    def __call__(self, Vdrv):
        Vdrv = Vdrv + self.__V_drv_noise
        Rhw = self.__wire.R
        I_measured = Vdrv / (self.__R_I + self.__R_w + Rhw)
        dVI_adc = self.__adc_I(((I_measured * self.__R_I) + self.__I_amp_noise) * self.__I_gain, return_type='voltage_adc', samples=self.__samples)
        I_drv = dVI_adc / (self.__I_gain * self.__R_I)
        dVv_adc = self.__adc_V(((I_measured * Rhw) + self.__V_amp_noise) * self.__V_gain, return_type='voltage_adc', samples=self.__samples)
        V_hw = dVv_adc / self.__V_gain
        Vdrv_adc = self.__adc_V_drv(Vdrv * self.__V_drv_gain, return_type='voltage_adc', samples=self.__samples)
        V_drv = Vdrv_adc / self.__V_drv_gain
        P_hw = nrvs.mean(V_hw) * nrvs.mean(I_measured)
        P_waste = nrvs.mean(I_measured)**2 * (self.__R_I + self.__R_w)
        P_total = nrvs.mean(Vdrv) * nrvs.mean(I_measured)
        efficiency = None
        if not P_total == 0:
            efficiency = 100 * (P_hw / P_total)

        data = {
            'V_drv': V_drv,
            'I_drv': I_measured,
            'V_hw': V_hw,
            'P_hw': P_hw,
            'P_waste': P_waste,
            'P_total': P_total,
            'P_RI': nrvs.mean(I_measured)**2 * self.__R_I,
            'efficiency': efficiency,
        }
        return V_drv, V_hw, I_drv, data


class HotwireDriver:
    def __init__(self, Vin, f_clk=72e6, f_pwm=100e3, C_filt=1000e-6, L_filt=22e-6):
        self.__Vin = Vin
        self.__f_clk = f_clk
        self.__f_pwm = f_pwm
        self.__pwm_counts = int(f_clk / f_pwm)
        self.__tau_filt = (2 * math.pi * math.sqrt(C_filt * L_filt))
        self.__f_filt = 1 / self.__tau_filt
        # print('Driver Corner Frequency = {} Hz'.format(self.__f_filt))
        # NOTE: This noise calculation is a major hack.  It would be nice to fix it.
        ripple_attenuation_ratio = 0.5 * pow(10, -math.log10(f_pwm / self.__f_filt))
        self.__V_ripple_rms = (Vin * ripple_attenuation_ratio / math.sqrt(2)) / 2 / 5
        self.__duty_cycle_min = 0.001
        self.__duty_cycle_max = 1
        self.__V_drv_last = nrvs.NRV(0)

    def __str__(self):
        fmt = 'Driver(fclk = {:0.2f} MHz, fpwm = {:0.2f} kHz, fcorner = {:0.2f} Hz)'
        return fmt.format(self.__f_clk / 1e6, self.__f_pwm / 1e3, self.__f_filt)

    @property
    def effective_bits(self):
        return math.log2(self.__pwm_counts)

    @property
    def Vdrv(self):
        return self.__V_drv_last

    def update(self, dt, Vdrv_req):
        data = {}
        # Constrain Drive Voltage
        Vdrv_req = max(0, min(Vdrv_req, self.__Vin))
        data['Vrequested'] = Vdrv_req
        # Quantitize Drive Voltage
        Vdrv_req, noise, quantitize_data = self.__quantitize_Vdrv(Vdrv_req)
        data['Vquantized'] = Vdrv_req
        data.update(quantitize_data)
        data['Vripple'] = self.__V_ripple_rms
        # Process Low-Pass Filter Effect
        dV = Vdrv_req - nrvs.mean(self.__V_drv_last)
        Vdrv_mean = nrvs.mean(self.__V_drv_last) + (dV * (1 - math.exp(-self.__f_filt * dt)))
        self.__V_drv_last = nrvs.NRV(Vdrv_mean, sd=noise)
        return self.__V_drv_last, data

    def __quantitize_Vdrv(self, Vdrv):
        data = {}
        duty_cycle = Vdrv / self.__Vin
        data['D_exact'] = duty_cycle
        duty_cycle = max(self.__duty_cycle_min, min(self.__duty_cycle_max, duty_cycle))
        data['D_constrained'] = duty_cycle
        counts = int((duty_cycle * self.__pwm_counts) + 0.5)
        data['counts'] = counts
        noise = 0 if ((duty_cycle == 0) or (duty_cycle == 1)) else self.__V_ripple_rms
        return self.__Vin * (counts / self.__pwm_counts), noise, data


class HotwireSystem:
    __T0 = 273.15

    def __init__(self, Vin, wire, controller_type, f_update=100, ctrl_kws={}, afe_kws={}, lim_kws={}, filt_kws={}, Tamb=None, Tset=None, debug=True):
        self.__debug = debug
        self.__Tamb = HotwireSystem.__T0 + 23 if Tamb is None else Tamb
        Tset = self.__Tamb if Tset is None else Tset
        self.__wire = wire
        self.__wire.T = self.__Tamb
        self.__afe = HotwireAFE(self.__wire, **afe_kws)
        self.__driver = HotwireDriver(Vin)
        if debug:
            print('Constructing Hotwire System:')
            print('\tVin = {:0.1f} V'.format(Vin))
            print('\tTamb = {:0.1f} K'.format(self.__Tamb))
            print('\t{}'.format(self.__wire))
            print('\t{}'.format(self.__driver))
            print('\tNOTE: NEED TO MAKE THE DRIVER LIMIT CURRENT ALSO')
            print('\n\tAFE Measurement Limits:')
        range_fmt = '\t\t{} -> ({:0.2f} {units}, {:0.2f} {units})'
        lim_kws = {'Tset': Tset}
        for name, (vmin, vmax) in self.__afe.ranges.items():
            vmin, vmax = nrvs.mean(vmin), nrvs.mean(vmax)
            units = 'V' if name[0] == 'V' else 'A'
            if name == 'V_drv':
                lim_kws['Vmax'] = vmax
            elif name == 'I_drv':
                lim_kws['Imax'] = vmax
            if debug:
                print(range_fmt.format(name, vmin, vmax, units=units))
        self.__controller = controller_type(Vin, self.__wire, self.__Tamb, **ctrl_kws, **filt_kws, **lim_kws)
        self.__controller.T = self.__Tamb
        if debug:
            print('\n\t{}'.format(self.__controller))
        self.__t = 0
        self.__dt = 1 / f_update
        self.__data = None
        self.__create_empty_data()

    @property
    def f_update(self):
        return 1 / self.__dt

    @property
    def dt(self):
        return self.__dt

    @property
    def wire(self):
        return self.__wire

    @property
    def afe(self):
        return self.__afe

    @property
    def driver(self):
        return self.__driver

    @property
    def controller(self):
        return self.__controller

    @property
    def t(self):
        return self.__t

    @property
    def Tamb(self):
        return self.__Tamb

    @Tamb.setter
    def Tamb(self, T):
        self.__Tamb = T
        return self.__Tamb

    @property
    def Tset(self):
        return self.__controller.Tset

    @Tset.setter
    def Tset(self, T):
        self.__controller.Tset = T
        return self.__controller.Tset

    @property
    def data(self):
        return self.__data  # TODO: THIS SHOULD RETURN A COPY AND AN OBJECT WITH THE DATA AS ATTRIBUTES

    @property
    def debug(self):
        return self.__debug

    def reset(self, Tinit=None, init_samples=500, settle_samples=10, V_init=0.5):
        Tinit = self.Tamb if Tinit is None else Tinit
        self.__wire.T = Tinit
        self.__controller.T = Tinit
        self.__controller.T_est.T0 = Tinit
        R0 = self.__measure_initial_resistance(init_samples, settle_samples, V_init)
        self.__controller.T_est.R0 = R0
        self.__controller.R_est.R0 = R0
        if(self.debug):
            print()
            print('Reset hotwire system with Tamb = {}K, measured the wire as {}ohm ({:0.4f}ohm actual)'.format(Tinit, R0, self.__wire.rFromT(Tinit)))
        self.__t = 0
        self.__create_empty_data()

    def __measure_initial_resistance(self, init_samples, settle_samples, V_init):
        R_sum = 0
        for idx in range(init_samples+settle_samples):
            _, V_hw, I_hw, _ = self.__afe(self.__driver.Vdrv)
            V_drv, _ = self.__driver.update(self.__dt, V_init)
            self.__wire.update(self.__dt, I_hw, Pload=0, Tamb=self.Tamb)
            if idx >= settle_samples:
                R_sum += (V_hw / I_hw)
        return R_sum / init_samples

    def update(self, dt, Pload=0, Pload_ff=0, V_drv_force=None):
        self.__data['ts'].append(self.__t)
        V_drv, V_hw, I_hw, afe_data = self.__afe(self.__driver.Vdrv)
        self.__data['afe_data'].append(afe_data)
        self.__data['Is'].append(afe_data['I_drv'])
        self.__data['Is_est'].append(I_hw)
        self.__data['Vs_hw_est'].append(V_hw)
        # effs.append(afe_data['efficiency'])
        # Ps_RI.append(afe_data['P_RI'])
        V_drv, ctrl_data = self.__controller.update(dt, V_drv, V_hw, I_hw, Pload=Pload_ff)
        if V_drv_force is not None:
            V_drv = V_drv_force
        self.__data['Ts_est'].append(ctrl_data['T_est'])
        self.__data['Ts_set'].append(self.__controller.Tset)
        # Vs_tgt.append(ctrl_data['Vtgt'])
        # Vs_correct.append(ctrl_data['Vcorrect'])
        self.__data['Rs_est'].append(ctrl_data['R_est'])
        V_drv, driver_data = self.__driver.update(dt, V_drv)
        self.__data['Vs_drv'].append(V_drv)
        # Vs_drv_requested.append(driver_data['Vrequested'])
        # Vs_drv_quantized.append(driver_data['Vquantized'])
        wire_data = self.__wire.update(dt, I_hw, Pload=Pload, Tamb=self.Tamb)
        self.__data['wire_data'].append(wire_data)
        self.__data['Rs'].append(wire_data['R'])
        self.__data['Ts'].append(wire_data['T'])
        # Vs_hw.append(V_hw)
        # loads.append(Pload)
        self.__t += self.__dt
        return self.__t

    def __create_empty_data(self):
        self.__data = {
            'ts': [],
            'Ts': [],
            'Ts_est': [],
            'Ts_set': [],
            'Rs': [],
            'Rs_est': [],
            'Is': [],
            'Is_est': [],
            'Vs_drv': [],
            'Vs_drv_est': [],
            'Vs_hw': [],
            'Vs_hw_est': [],
            'cut_data': [],
            'afe_data': [],
            'driver_data': [],
            'controller_data': [],
            'wire_data': []
        }


class HotwireSimulator:
    def __init__(self, t_max=90, cut_sim=None, seed=None):
        self.__t_max = t_max
        self.__cut_sim = HotwireCutSimulation() if cut_sim is None else cut_sim
        self.__seed = seed

    def run_sim(self, hw_system, Tset, Tinit=None, test_ff=True):
        if self.__seed is not None:
            random.seed(self.__seed)
        hw_system.reset(Tinit=Tinit)
        self.__cut_sim.reset(Tinit=Tinit)
        dt = hw_system.dt
        hw_system.Tset = Tset
        while hw_system.t < self.__t_max:
            Pload = self.__cut_sim(hw_system.t, hw_system.wire)
            Pload += nrvs.NRV.Noise(variance=0.1*Pload)
            Pload_ff = Pload if test_ff else 0
            hw_system.update(dt, Pload=Pload, Pload_ff=Pload_ff)
            # TODO: RECORD CUT PROPERTIES, SPEED, WIDTH, VOLUME, KERF, ETC.
        return hw_system.data


if __name__ == '__main__':
    f_update = 100
    # dt = 1 / f_update
    Vin = 36
    # Vmax = None
    # Imax = None
    T0 = 273.15
    Tamb = T0 + 30
    Tset = T0 + 315
    # wire, pid_kws = HotWire('Nifethal 70', 30, 1.3), {'Kp': 0.8, 'Ki': 0.02}
    # wire, pid_kws = HotWire('316L', 30, 1.3), {'Kp': 2, 'Ki': 0.002, 'Kd': 0.075}  # {'Kp': 2.5, 'Ki': 0.01}
    # wire, pid_kws = HotWire('Nikrothal 60', 30, 1.3), {'Kp': 0.5, 'Ki': 0.02}
    # wire, pid_kws = HotWire('Nikrothal 80', 30, 1.3), {'Kp': 0.4, 'Ki': 0.015}
    # wire, pid_kws = HotWire('Kanthal A1', 28, 1.3), {'Kp': 0.25, 'Ki': 0.001, 'Kd': 0.025}  # {'Kp': 0.45, 'Ki': 0.015}
    wire, pid_kws = HotWire('Ni200', 30, 1.3), {'Kp': 0.7, 'Ki': 0.02}
    # wire, pid_kws = HotWire('Nifethal 52', 30, 1.3), {'Kp': 1.0, 'Ki': 0.02}
    # pid_kws = {'Kp': 0.1, 'Ki': 0.001, 'Kd': 0.01}

    hw_sys = HotwireSystem(Vin, wire, PredictiveHotwireController, ctrl_kws=pid_kws, f_update=f_update)
    # hw_sys.reset()


    hw_sim = HotwireSimulator()
    sim_data = hw_sim.run_sim(hw_sys, Tset=Tset, Tinit=Tamb)
    # print(sim_data)
    ts = np.array(sim_data['ts'])
    Ts = np.array([nrvs.mean(T) for T in sim_data['Ts']])
    Rs = np.array([nrvs.mean(R) for R in sim_data['Rs']])
    Ts_est = np.array([nrvs.mean(T) for T in sim_data['Ts_est']])
    Ts_est_sd = np.array([nrvs.standard_deviation(T) for T in sim_data['Ts_est']])
    Ts_set = np.array([nrvs.mean(T) for T in sim_data['Ts_set']])

    Is = np.array([nrvs.mean(I) for I in sim_data['Is']])

    fig, axs = plt.subplots(2, figsize=(16, 9), sharex=True, constrained_layout=True)
    Tamb = hw_sys.Tamb
    axs[0].scatter(ts, Ts_est-T0, alpha=0.75, color='c', s=1)
    axs[0].errorbar(ts, Ts_est-T0, yerr=3*Ts_est_sd, alpha=0.01, color='c', fmt=' ', zorder=-1)
    axs[0].plot(ts, Ts-T0, alpha=1, c='b')
    axs[0].plot(ts, Ts_set-T0, alpha=1, c='k')
    print(Ts_est_sd)
    axs[0].axhline(Tamb-T0, c='r')
    g = axs[1].plot(ts, Rs)
    axs[1].yaxis.label.set_color(g[-1].get_color())
    axs[1].tick_params(axis='y', colors=g[-1].get_color())
    axs[1].set_ylabel(r'Resistance ($\Omega$)')
    # axs[1].set_ylim(0, 25)
    ax1 = axs[1].twinx()
    ax1.plot(ts, Is, c='g')
    ax1.yaxis.label.set_color('g')
    ax1.tick_params(axis='y', colors='g')
    ax1.set_ylabel(r'Current (A)')

    plt.show()
