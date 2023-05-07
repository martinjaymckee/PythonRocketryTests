import numpy as np
import matplotlib.pyplot as plt

Lb = -0.0065 # K m
R = 8.3144598 # (N m) / (mol K)
g0 = 9.80665 # m s^-2
M = 0.0289644 # kg mol^-1
C = ((-R*Lb)/((g0*M)))

def dhdTb(P, Pb, Tb):
    global C

    return ( pow(P/Pb, C) - 1) / Lb

def dhdPb(P, Pb, Tb):
    global g0
    global M
    global C
    denom = R * Tb * pow(P/Pb, C)
    num = M * Pb * g0
    return denom / num    

def pressure(h, Pb=1013250, Tb=288.15, hb = 0):
    global Lb
    global C
    return Pb * pow(((Tb + ((h - hb) * Lb)) / Tb), 1/C)

if __name__ == '__main__':
    N = 5
    M = 10
    h_max = 3000
    T_ref = 273.15
    Tb_range = (T_ref + 0, T_ref + 40)
    Pb_range = (87000, 108500)
    Tbs = np.linspace(*Tb_range, N)
    Pbs = np.linspace(*Pb_range, N)
    hs = np.linspace(0, h_max, M)
    fig, axs = plt.subplots(M, sharex = True, constrained_layout=True)
    
    for idx, h in enumerate(hs):
        P = pressure(h)
        #axs[idx].set_title('P = {} Pa'.format(P))
        
        for Tb in Tbs:
            divs = []
            for Pb in Pbs:
                divs.append(dhdTb(P, Pb, Tb)) # + dhdPb(P, Pb, Tb))
            divs = np.array(divs)
            axs[idx].plot(Tbs, divs, label='{}'.format(Tb))
        #axs[idx].legend()
    
    plt.show()