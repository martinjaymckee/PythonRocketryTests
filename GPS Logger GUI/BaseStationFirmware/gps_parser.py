import crc16


class GPSSample:
    def __init__(self):
        self.id = None
        self.rssi = 0
        self.remote_sats = 0
        self.remote_pos = [0]*3
        self.remote_vel = [0]*3
        self.base_sats = 0
        self.base_pos = [0]*3
        self.__crc_processor = crc16.CRC16_CCITT_False()
        
    @property
    def valid(self):
        return self.id is not None
    
    @property
    def stream_format(self):
        if self.valid:
            self.__crc_processor.reset()
            text = '{:d},'.format(int(self.id))
            text += '{:d},'.format(int(self.rssi))
            
            # Remote GPS
            text += '{:d},'.format(int(self.remote_sats))
            text += '{:0.9f},{:0.9f},{:0.9f},'.format(*self.remote_pos)
            text += '{:0.9f},{:0.9f},{:0.9f},'.format(*self.remote_vel)
            
            # Base GPS
            text += '{:d},'.format(int(gauss(self.__base_sats, self.__sats_sd)))
            text += '{:0.9f},{:0.9f},{:0.9f},'.format(*self.remote_pos)

            crc_processor = crc16.CRC16_CCITT_False()
            crc = crc_processor(bytes(text, 'ascii'))
            text += '0x{:04X}\n'.format(crc)
        raise Exception('GPSSample was not valid when trying to create stream format')
    
    
    def __str__(self):
        text = 'GPS( {:d} dB'.format(self.rssi)
        text += ', remote sats = {:d}'.format(self.remote_sats)
        text += ', remote xyz = [{:0.3f}, {:0.3f}, {:0.3f}]'.format(*self.remote_pos)
        text += ', remote vel = [{:0.3f}, {:0.3f}, {:0.3f}]'.format(*self.remote_vel)        
        text += ', base sats = {:d}'.format(self.base_sats)
        text += ', base xyz = [{:0.3f}, {:0.3f}, {:0.3f}] )'.format(*self.remote_pos)
        return text
    
    
class GPSParser:
    class ParserStatistics:
        def __init__(self, debug=False):
            self.reset()
            self.__debug = debug
        
        def __str__(self):
            return 'Parser Statistics: {} samples, {} segments, {} errors'.format(self.samples, self.segments, self.errors)
        
        def reset(self):
            self.samples = 0
            self.segments = 0
            self.errors = 0
            
    def __init__(self, debug=True):
        self.__buffer = ''
        self.__last_gps_sample = GPSSample()
        self.__crc_processor = crc16.CRC16_CCITT_False()
        self.__parser_statistics = GPSParser.ParserStatistics()
        self.__debug = debug
        
    @property
    def statistics(self):
        return self.__parser_statistics

    def __call__(self, data):
        self.__buffer += data
            
        done = False
        results = []
        while not done:
            idx = self.__buffer.find('\n')
            if idx < 0:
                done = True
            else:
                segment = self.__buffer[:idx]
                self.__buffer = self.__buffer[idx+1:]  # Copy remainder except for the '\n'
                self.__parser_statistics.segments += 1
                crc_valid = False
                crc_idx = segment.rfind(',')
                if not crc_idx == -1:
                    try:
                        self.__crc_processor.reset()
                        calculated_crc = self.__crc_processor(bytes(segment[:crc_idx+1], 'ascii'))
                        read_crc = int(segment[crc_idx+1:], 16)
                        crc_valid = (calculated_crc == read_crc)
                    except Exception as e:
                        if self.__debug:
                            print('Character Encoding Error')
                if crc_valid:
                    tokens = [t.strip() for t in segment.split(',')]
                    gps_sample = GPSSample()
                    try:
                        gps_sample.id = int(tokens[0])
                        if self.__last_gps_sample.id is None:
                            self.__last_gps_sample.id = gps_sample.id
                        elif gps_sample.id == self.__last_gps_sample.id:
                            if self.__debug:
                                print('Repeated Sample')
                            break
                        elif gps_sample.id > (self.__last_gps_sample.id+1):
                            offset = (gps_sample.id - self.__last_gps_sample.id)
                            if self.__debug:
                                print('Missed {} Frames'.format(offset))
                            self.__parser_statistics.errors += offset

                        gps_sample.rssi = int(tokens[1])
                        gps_sample.remote_sats = int(tokens[2])
                        gps_sample.remote_pos = [float(v) for v in tokens[3:6]]                        
                        gps_sample.remote_vel = [float(v) for v in tokens[6:9]]                        
                        gps_sample.base_sats = int(tokens[9])
                        gps_sample.base_pos = [float(v) for v in tokens[10:13]]                                                
                        
                        self.__parser_statistics.samples += 1                        
                        results.append(gps_sample)
                        self.__last_gps_sample = gps_sample
                    except Exception as e:
                        if self.__debug:
                            print('Parsing Error - {}'.format(e))
                        self.__parser_statistics.errors += 1                                            
                else:
                    if self.__debug:
                        print('CRC Invalid')      
                    self.__parser_statistics.errors += 1
        return results