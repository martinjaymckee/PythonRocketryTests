import electronics.discretes as discrete


def optimalADCDivider(Vadc, Vin_max, Zadc):
    Ztgt = Zadc / 10
    Ra = discrete.Resistor.floor((Ztgt*Vin_max) / Vadc)
    Rb = discrete.Resistor.closest((Vadc*Ra) / (Vin_max - Vadc))
    return Ra, Rb


def ratio(Ra, Rb):
    return Rb / (Ra + Rb)


def parallel(Za, Zb):
    return Za*Zb / (Za + Zb)


def divider_error_percent(Vadc, Vin_max, Ra, Rb):
    Vadc_test = Vin_max * ratio(Ra, Rb)
    return 100 * abs(Vadc - Vadc_test) / Vadc


if __name__ == '__main__':
    Vadc = 3.3
    Vin_max = 8.4
    e = 0.05
    Zadc = 0.1e6
    Vin_max *= (1 + e)
    Ra, Rb = optimalADCDivider(Vadc, Vin_max, Zadc)
    print('Vadc = {} V, Vin_max = {} V'.format(Vadc, Vin_max))
    print('Ra = {} ohms, Rb = {} ohms, Ra||Rb = {} ohms'.format(Ra, Rb, parallel(Ra, Rb)))
    print('ratio = {}, percent error = {} %'.format(ratio(Ra, Rb), divider_error_percent(Vadc, Vin_max, Ra, Rb)))
