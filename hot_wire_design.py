import math


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


if __name__ == '__main__':
    L = 1
    awg = 26
    P = 50
    print('Diameter {}AWG = {:0.3f} mm'.format(awg, wireDiametermm(awg)))
    print('Resistance {:d} mm, {} AWG = {:0.3f} ohm'.format(int(L * 1000), awg, wireResistance(L, awg)))
    V, I = driverCharacteristics(L, awg, P)
    print('Drive Characterstics {} W = {:0.2f} V, {:0.2f} A'.format(P, V, I))

    awg = 24
    R = wireResistance(1, awg, 1.12e-6)
    P = 50
    V = 12
    L = V**2 / (P * R)
    print('For a {:0.1f} W, {:0.1f} V heater of {:d} awg Nichrome, {:0.2f} m is needed.'.format(P, V, awg, L))
