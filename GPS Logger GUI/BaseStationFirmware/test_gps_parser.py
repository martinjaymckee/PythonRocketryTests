import crc16
import gps_parser

    
    
if __name__ == '__main__':
    import random

    def gauss(mean, sd, N=12):
        t = 0
        for _ in range(N):
            t += random.uniform(-1, 1)
        return mean + (sd * t / float(N))

    class VirtualGPSGenerator:
        def __init__(self):
            self.__default_remote_pos = [35, -105, 4500]
            self.__dt = 0.2
            self.__id = 0
            self.__remote_sats = 9
            self.__sats_sd = 2
            self.__rssi = -123
            self.__rssi_sd = 5.5
            self.__remote_pos = self.__default_remote_pos
            self.__remote_vel = (0, 0, -5)
            self.__base_sats = 11
            self.__base_pos = (35, -105, 2300)
            self.__lat_lon_sd = 1e-3
            self.__alt_sd = 1
            
            self.__last_gps_line = None
            
            self.__generated = []
            
        def reset(self):
            self.__id = 0
            self.__remote_pos = self.__default_remote_pos
            self.__last_gps_line = None
            self.__generated = []
            return
        
        def last(self):
            return self.__last_gps_line

        def next(self):
            line = '{:d},'.format(int(self.__id))
            line += '{:d},'.format(int(gauss(self.__rssi, self.__rssi_sd)))
            
            # Remote GPS
            line += '{:d},'.format(int(gauss(self.__remote_sats, self.__sats_sd)))
            line += '{:0.9f},'.format(gauss(self.__remote_pos[0], self.__lat_lon_sd))
            line += '{:0.9f},'.format(gauss(self.__remote_pos[1], self.__lat_lon_sd))
            line += '{:0.9f},'.format(gauss(self.__remote_pos[2], self.__alt_sd))
            line += '{:0.9f},'.format(gauss(self.__remote_vel[0], self.__lat_lon_sd))
            line += '{:0.9f},'.format(gauss(self.__remote_vel[1], self.__lat_lon_sd))
            line += '{:0.9f},'.format(gauss(self.__remote_vel[2], self.__alt_sd))
            
            # Base GPS
            line += '{:d},'.format(int(gauss(self.__base_sats, self.__sats_sd)))
            line += '{:0.9f},'.format(gauss(self.__base_pos[0], self.__lat_lon_sd))
            line += '{:0.9f},'.format(gauss(self.__base_pos[1], self.__lat_lon_sd))
            line += '{:0.9f},'.format(gauss(self.__base_pos[2], self.__alt_sd))            

            crc_processor = crc16.CRC16_CCITT_False()
            crc = crc_processor(bytes(line, 'ascii'))
            line += '0x{:04X}\n'.format(crc)

            if self.__remote_pos[2] > self.__base_pos[2]:
                self.__remote_pos[0] += self.__remote_vel[0]
                self.__remote_pos[1] += self.__remote_vel[1]
                self.__remote_pos[2] += self.__remote_vel[2]
                
            self.__last_gps_line = line
            self.__id += 1
            return line
            
    gen = VirtualGPSGenerator()
    parser = gps_parser.GPSParser()
    
    results = []
    for _ in range(10):
        line = gen.next()
        print(line)
        new_results = parser(line)
        results += new_results
        
    print(parser.statistics)
    
    
