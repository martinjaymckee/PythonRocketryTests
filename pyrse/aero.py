import math

class StandardISAConditions:
    P = 101325 # Pa - Air Pressure
    T = 288.15 # K - Air Temperature
    rho = 1.225 # kg/m^3 - Density
    mu = 1.81e-5 # Pa-s - Dynamic Viscosity
    nu = 1.48e-5 # m^2/s - Kinematic Viscosity


def Re(V, L, rho=None, mu=None):
    rho = rho if rho is not None else StandardISAConditions.rho
    mu = mu if mu is not None else StandardISAConditions.mu
    return (rho * V * L) / mu


def mach(V, T=None):
    T = T if T is not None else StandardISAConditions.T
    R = 287.05 # J/kg-K
    a = math.sqrt(1.4*R*T)
    return V/a

