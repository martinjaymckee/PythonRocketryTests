import math

import pyrse.aero as aero_utils

class Airfoil:
    def __init__(self):
        pass

    def Cl(self, alpha, Re=None):
        assert True, 'Error: {}.Cl(...) is not implemented!'.format(self.__class__.__name__)

    def Cd(self, alpha, Re=None):
        assert True, 'Error: {}.Cd(...) is not implemented!'.format(self.__class__.__name__)

    def Cm(self, alpha, Re=None):
        assert True, 'Error: {}.Cm(...) is not implemented!'.format(self.__class__.__name__)


class FlatPlateAirfoil(Airfoil):
    def __init__(self, thickness=None, alpha_stall=math.radians(12), cl_stall=0.7, k_cl=0.5, k_cd_re=0.15, k_cd_t=2.0, Cd_0=0.1):
        self.__thickness = thickness
        self.__alpha_stall = alpha_stall
        self.__cl_stall = cl_stall
        self.__k_cl = k_cl
        self.__k_cd_re = k_cd_re
        self.__k_cd_t = k_cd_t        
        self.__Cd_0 = Cd_0


    def Cl(self, alpha, Re=None):
        cl_base = None
        if alpha > self.__alpha_stall:
            cl_base = self.__cl_stall
        elif alpha < -self.__alpha_stall:
            cl_base = - self.__cl_stall
        else:
            cl_base = 2*math.pi*alpha
        return cl_base * (1 - (self.__k_cl * self.__thickness))


    def Cd(self, alpha, Re=None):
        Cd_base = 1.28 * math.sin(alpha)**2 + self.__Cd_0
        Cd_thin = Cd_base * ( 1 + (self.__k_cd_re / math.sqrt(Re))) if (Re is not None) and (Re > 100) else Cd_base
        Cd_tot = Cd_thin + (self.__k_cd_t*self.__thickness) if self.__thickness is not None else Cd_thin
        return Cd_tot
    
    def Cm(self, alpha, Re=None):
        return -0.15 # HACK: This is just a guess.