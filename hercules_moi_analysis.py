import math

test_data = [
    {
        'name': 'Hercules (1/2 stick)',
        'l': 0.6842,
        'r': 0.33,
        'm': 0.6324,
        'cg': 0.456,
        'cp': {'rocksim': 0.5095, 'barrowman': 0.4640},
        'S_fin': 0.0210,
        'N_fin': 4,
        'ts': [27.64, 28.69, 28.6, 28.43, 28.83]
    }, 
    {
        'name': 'Hercules (1 stick)',
        'l': 0.6842,
        'r': 0.33,
        'm': 0.6324,
        'cg': 0.456,
        'cp': {'rocksim': 0.5095, 'barrowman': 0.4640},
        'S_fin': 0.0210,
        'N_fin': 4,
        'ts': [27.64, 28.69, 28.6, 28.43, 28.83],
        'm_add': 0.063,
        'm_loc': 0.1
    },     
    {
        'name': 'Catalyst',
        'l': 0.6953,
        'r': 0.235,
        'm': 0.5224,
        'cg': 0.594,
        'cp': {'rocksim': 0.7079, 'barrowman': 0.6844},        
        'S_fin': 0.0091,
        'N_fin': 4,
        'ts': [42.04, 42.89, 41.59, 41.99, 41.75]
    }, 
    {
        'name': 'Hiroc',
        'l': 0.7144,
        'r': 0.29,
        'm': 0.5942,
        'cg': 0.694,
        'cp': {'rocksim': 0.8055, 'barrowman': 0.7532},
        'S_fin': 0.0103,
        'N_fin': 4,
        'ts': [40.08, 40.13, 40.23, 40.19, 40.09]
    }, 
    {
        'name': 'Onyx',
        'l': 0.6715,
        'r': 0.215,
        'm': 0.4644,
        'cg': 0.432,
        'cp': {'rocksim': 0.4980, 'barrowman': 0.4581},
        'S_fin': 0.008,
        'N_fin': 3,
        'ts': [34.12, 34.59, 34.28, 34.40, 34.38]
    }, 
    {
        'name': 'Patriot',
        'l': 0.7207,
        'r': 0.305,
        'm': 0.5325,
        'cg': 0.562,
        'cp': {'rocksim': 0.6247, 'barrowman': 0.5687},
        'S_fin': 0.0055,
        'N_fin': 4,
        'ts': [36.56, 36.23, 36.49, 36.46, 36.53]
    }                
]

class MOITest:
    def __init__(self, l, r, m, ts, m_add=[], cycles=25, g=9.80665):
        self.__l = l
        self.__r = r
        self.__m = m
        self.__ts = ts
        self.__m_add = m_add
        self.__cycles = 25
        self.__g = g
        self.__t_mean = None
        self.__t_sd = None
        self.__base_moi = 0
        self.__mois = []
        self.__moi_mean = None
        self.__moi_sd = None
        self.__update_mois()
        self.__update_stats()

    def __str__(self):
        return 'MOITest(l = {:0.2g} m, r = {:0.2g} m, m = {:0.2g} kg, t = {:0.2g} (+/- {:0.2g}) s, moi = {:0.2g} (+/- {:0.2g}) kg-m^2)'.format(self.__l, self.__r, self.__m, self.__t_mean, self.__t_sd, self.__moi_mean, self.__moi_sd)

    @property
    def t(self):
        return self.__t_mean

    @property
    def t_sd(self):
        return self.__t_sd

    @property
    def moi(self):
        return self.__moi_mean

    @property
    def moi_sd(self):
        return self.__moi_sd

    def __update_mois(self):
        self.__base_moi = 0
        for r, m in self.__m_add:
            self.__base_moi += m * (r**2)
        self.__mois = []
        for t in self.__ts:
            t /= self.__cycles
            moi = ((self.__m * self.__g) / self.__l) * (((t * self.__r) / (2 * math.pi))**2)
            self.__mois.append(moi+self.__base_moi)
    
    def __update_stats(self):
        if (self.__mois is None) or (not len(self.__ts) == len(self.__mois)):
            self.__update_mois()
        self.__t_mean = 0
        self.__moi_mean = 0
        N = len(self.__ts)
        ts = []
        for t, moi in zip(self.__ts, self.__mois):
            t /= self.__cycles
            ts.append(t)
            self.__t_mean += t
            self.__moi_mean += moi
        self.__t_mean /= N
        self.__moi_mean /= N
        self.__t_sd = 0
        self.__moi_sd = 0
        for t, moi in zip(ts, self.__mois):
            self.__t_sd += ((t - self.__t_mean)**2)
            self.__moi_sd += ((moi - self.__moi_mean)**2)
        self.__t_sd = math.sqrt(self.__t_sd / (N - 1))            
        self.__moi_sd = math.sqrt(self.__moi_sd / (N - 1))                    

if __name__ == '__main__':
    filename = 'hercules_moi_analysis.dat'
    with open(filename, 'w') as file:
        for data in test_data:
            cg = data['cg']
            m = data['m']
            moi_m_add = []
            if 'm_add' in data:
                m_add = data['m_add'] if 'm_add' in data else 0
                m_loc = data['m_loc'] if 'm_loc' in data else 0
                cg_new = ((cg * m) + (m_add * m_loc)) / (m + m_add)
                m += m_add
                cg_offset = cg - cg_new
                #print('*** CG offset by {:0.1f} mm'.format(1000*cg_offset))
                cg = cg_new 
                loc = m_loc - cg
                moi_m_add.append((loc, m_add))
            moi_test = MOITest(data['l'], data['r'], data['m'], data['ts'], m_add=moi_m_add)                
            file.write('{} -> {}\n'.format(data['name'], moi_test))
            file.write('\tTotal Mass: {:0.3f} kg\n'.format(m))
            file.write('\tCG: {:0.3f} m\n'.format(cg))
            moi = moi_test.moi
            file.write('\tMOI: {:0.3f} kg m^2\n'.format(moi))            
            for eq, cp in data['cp'].items():
                sm = cp - cg
                fin_volume = data['S_fin'] * data['N_fin'] * sm / 2
                file.write('\tWith {} CP @ {:0.3f} m:\n'.format(eq, cp))
                file.write('\t\tStatic Margin: {:0.1f} mm ({:0.2f}% of cg)\n'.format(sm*1000, 100*sm/cg))
                file.write('\t\tFin Volume (Vf): {:0.2f} cm^3\n'.format(fin_volume*100*100*100))
                file.write('\t\tFin Power ((m*(Vf^2/3)/I)): {:0.4f}\n'.format((data['m'] * (fin_volume**0.667) / moi))) 
                file.write('\n')

