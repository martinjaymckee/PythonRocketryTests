import matplotlib.pyplot as plt
import numpy as np


def calcVset(Vin, Rset, Ra, Rb, R_scale):
    B = Rb / (Ra + Rb)
    C = R_scale
    return (B * Vin) / (C * Rset)
    

if __name__ == '__main__':
    R_range = (0.5, 1000)
    V_range = (3, 15)
    V_ref = 2.5
    
    Imax = max(V_range) / min(R_range)
    
    Av = 20
    Rs = (V_ref / Av) / Imax
    P_Rs = (max(V_range) ** 2) * Rs
    
    Ra = 5000
    Rb = 1000
    
    R_scale = 1 #V_ref / max(R_range)
    print(Rs)
    print(P_Rs)
    print(R_scale)

    Vs = np.linspace(*V_range, 11)
    Rs = np.linspace(*R_range, 25)
    fig, ax = plt.subplots(1, constrained_layout=True)
    
    
    for V in Vs:
        Vsets = []
        for R in Rs:
            Vsets.append(calcVset(V, R, Ra, Rb, R_scale))
        Vsets = np.array(Vsets)
        Isets = Vsets / (Av * Rs)
        Rsets = V / Isets
        ax.plot(Rs, Rsets)
        
#    ax.set_ylim(0, V_ref)
    plt.show()
        