# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 00:26:46 2022

@author: marti
"""

def crc16_ccitt_false(data, crc=0xFFFF):
    t = 0
    for datum in data:
        t = (crc >> 8) ^ datum
        t = (t ^ (t >> 4))
        crc = (crc << 8) ^ (t << 12) ^ (t << 5) ^ t
        crc &= 0xFFFF
    return crc


import random
class VirtualLogGenerator:
    def __init__(self):
        self.__sample = 0
        self.__temp = 30
        self.__temp_sd = 0.1
        self.__vsense = 6.7
        self.__vsense_sd = 0.05
        self.__page_status = 0xFA
        self.__rssi = -123
        self.__rssi_sd = 1.5
        self.__logger_pos = (35, -105, 2700)
        self.__base_pos = (35, -105, 2500)
        self.__lat_lon_sd = 1e-3
        self.__alt_sd = 1
        self.__accel = (-9.81, 0, 0)
        self.__accel_sd = 0.01
        self.__gyro = (0, 0, 0)
        self.__gyro_sd = 0.1
        self.__mag = (1200, 900, -150)
        self.__mag_sd = 5.5
        self.__external_status = 0x03
        
        self.__last_log_line = None
        
    def last(self):
        return self.__last_log_line

    def next(self):  # TODO: NEED TO ADD ERRORS TO THE GENERATED OUTPUT
        line = '{},'.format(self.__sample)
        if (self.__sample % 10) == 0:
            line += '{:0.4f},'.format(random.gauss(self.__temp, self.__temp_sd))
            line += '{:0.4f},'.format(random.gauss(self.__vsense, self.__vsense_sd))
            line += '0x{:02X},'.format(self.__page_status)
            line += '{:d},'.format(int(random.gauss(self.__rssi, self.__rssi_sd)))            
            
            # Logger Pos
            line += '{:0.9f},'.format(random.gauss(self.__logger_pos[0], self.__lat_lon_sd))
            line += '{:0.9f},'.format(random.gauss(self.__logger_pos[1], self.__lat_lon_sd))
            line += '{:0.9f},'.format(random.gauss(self.__logger_pos[2], self.__alt_sd))
            
            # Base Pos
            line += '{:0.9f},'.format(random.gauss(self.__base_pos[0], self.__lat_lon_sd))
            line += '{:0.9f},'.format(random.gauss(self.__base_pos[1], self.__lat_lon_sd))
            line += '{:0.9f},'.format(random.gauss(self.__base_pos[2], self.__alt_sd))            
        else:
            line += ',,,,,,,,,,'
        # Accelerations
        line += '{:0.4f},'.format(random.gauss(self.__accel[0], self.__accel_sd))
        line += '{:0.4f},'.format(random.gauss(self.__accel[1], self.__accel_sd))
        line += '{:0.4f},'.format(random.gauss(self.__accel[2], self.__accel_sd))                    
        
        # Gyro Rates
        line += '{:0.4f},'.format(random.gauss(self.__gyro[0], self.__gyro_sd))
        line += '{:0.4f},'.format(random.gauss(self.__gyro[1], self.__gyro_sd))
        line += '{:0.4f},'.format(random.gauss(self.__gyro[2], self.__gyro_sd))                    

        # Magnetometer Readings
        line += '{:0.4f},'.format(random.gauss(self.__mag[0], self.__mag_sd))
        line += '{:0.4f},'.format(random.gauss(self.__mag[1], self.__mag_sd))
        line += '{:0.4f},'.format(random.gauss(self.__mag[2], self.__mag_sd))                    
        
        # External Status
        line += '0x{:02X},'.format(self.__external_status)        
        
        crc = crc16_ccitt_false(bytes(line, 'ascii'))
        line += '0x{:04X}\n'.format(crc)

        self.__sample += 1
        self.__last_log_line = line
        
        return line
        

class VirtualCommChannel:
    def __init__(self):
        pass
        # TODO: IDEALLY THIS OBJECT SHOULD GENERATE BURST ERRORS, BUT IT PROBABLY
        #   IS FINE WITH IT AS IT IS FOR BASIC TESTING 
    
    def __call__(self, text):
        if text is None:
            print('Passed in None to VirtualCommChannel')
            return ''
        p = 0.001
        new_text = ''
        for c in text:
            if random.uniform(0, 1) < p:
                c = chr(int(random.uniform(0, 255)))
            new_text += c
        return new_text
    

class LogSample:
    def __init__(self):
        self.sequence_id = None
        self.temperature = None
        self.battery_voltage = None
        self.page_status = None
        self.rssi = None
        self.logger_pos = [None] * 3
        self.base_pos = [None] * 3
        self.accels = [None] * 3
        self.gyros = [None] * 3
        self.mags = [None] * 3
        self.external_status = None
        
    def __str__(self):
        try:
            text = '{:d},'.format(self.sequence_id)
            text += '{:0.3f},'.format(self.temperature)
            text += '{:0.3f},'.format(self.battery_voltage)
            text += '0x{:02X},'.format(self.page_status)
            text += '{:d},'.format(self.rssi)
            text += '{:0.5f},{:0.5f},{:0.5f},'.format(*self.logger_pos)
            text += '{:0.5f},{:0.5f},{:0.5f},'.format(*self.base_pos)
            text += '{:0.5f},{:0.5f},{:0.5f},'.format(*self.accels)
            text += '{:0.5f},{:0.5f},{:0.5f},'.format(*self.gyros)
            text += '{:0.5f},{:0.5f},{:0.5f},'.format(*self.mags)
            text += '0x{:02X},'.format(self.external_status)
            crc = crc16_ccitt_false(bytes(text, 'ascii')) 
            text += '0x{:04X}'.format(crc)
            return text
        except Exception:
            return '<invalid>'
    
class LogParser:
    class ParserStatistics:
        def __init__(self):
            self.reset()
        
        def __str__(self):
            return 'Parser Statistics: {} samples, {} segments, {} errors'.format(self.samples, self.segments, self.errors)
        def reset(self):
            self.samples = 0
            self.segments = 0
            self.errors = 0
            
    def __init__(self, event_handler):
        self.__last_line = None
        self.__buffer = ''
        self.__event_handler = event_handler
        self.__last_log_sample = LogSample()
        self.__parser_statistics = LogParser.ParserStatistics()
        
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
                crc_valid = True  # TODO: RUN A CRC CHECK 
                if crc_valid:
#                    print(segment)
                    tokens = [t.strip() for t in segment.split(',')]
#                    print('tokens -> {}'.format(tokens))
                    log_sample = LogSample()
                    try:
                        log_sample.sequence_id = int(tokens[0])
                        if self.__last_log_sample.sequence_id is None:
                            self.__last_log_sample.sequence_id = log_sample.sequence_id
                        elif log_sample.sequence_id == self.__last_log_sample.sequence_id:
                            # NOTE: IF THE SAMPLE IS THE SAME AS THE LAST CORRECTLY PARSED
                            #   SAMPLE, THEN SIMPLY SEND AN ACK AND CONTINUE TO PARSING
                            #   THE NEXT SAMPLE.
                            print('Repeated Sample')
                            if self.__event_handler is not None:
                                self.__event_handler.ack()
                                break
                        elif log_sample.sequence_id > (self.__last_log_sample.sequence_id+1):
                            print('Invalid Sequence Number')
                            # TODO: NEED TO FIGURE OUT HOW TO HANDLE THIS SITUATION AS THE
                            #   CURRENT PROTOCOL DOESN'T HAVE ANY WAY TO TELL THE TRANSMITTER
                            #   THAT IT IS RUNNING AHEAD OF THE PARSER
                            self.__parser_statistics.errors += 1
                            if self.__event_handler is not None:
                                self.__event_handler.nack()
                                break

                        log_sample.temperature = self.__update_value(float, tokens[1], 'temperature')
                        log_sample.battery_voltage = self.__update_value(float, tokens[2], 'battery_voltage')
                        log_sample.page_status = self.__update_value(lambda x: int(x, 16), tokens[3], 'page_status')
                        log_sample.rssi = self.__update_value(int, tokens[4], 'rssi')
                        
                        log_sample.logger_pos[0] = self.__update_value(float, tokens[5], 'logger_pos', 0)                        
                        log_sample.logger_pos[1] = self.__update_value(float, tokens[6], 'logger_pos', 1)
                        log_sample.logger_pos[2] = self.__update_value(float, tokens[7], 'logger_pos', 2)
                        
                        log_sample.base_pos[0] = self.__update_value(float, tokens[8], 'base_pos', 0)                        
                        log_sample.base_pos[1] = self.__update_value(float, tokens[9], 'base_pos', 1)
                        log_sample.base_pos[2] = self.__update_value(float, tokens[10], 'base_pos', 2)
                        
                        log_sample.accels[0] = float(tokens[11])
                        log_sample.accels[1] = float(tokens[12])
                        log_sample.accels[2] = float(tokens[13])
                        
                        log_sample.gyros[0] = float(tokens[14])
                        log_sample.gyros[1] = float(tokens[15])
                        log_sample.gyros[2] = float(tokens[16])
                        
                        log_sample.mags[0] = float(tokens[17])
                        log_sample.mags[1] = float(tokens[18])
                        log_sample.mags[2] = float(tokens[19])
                        
                        log_sample.external_status = int(tokens[20], 16)
                        
#                        print(log_sample)
#                        print()
                        
                        self.__parser_statistics.samples += 1
                        if self.__event_handler is not None:
                            self.__event_handler.ack()
                        
                        results.append(log_sample)
                        self.__last_log_sample = log_sample
                    except Exception as e:
                        print('Parsing Error - {}'.format(e))
                        self.__parser_statistics.errors += 1                        
                        if self.__event_handler is not None:
                            print('Parsing Error')
                            self.__event_handler.nack()
                    
                else:
                    print('CRC Invalid')      
                    self.__parser_statistics.errors += 1                    
                    if self.__event_handler is not None:
                        self.__event_handler.nack()
        return results
      
    def __update_value(self, converter, token, key, index=None):
        new_value = None
        if len(token) == 0:
            new_value = getattr(self.__last_log_sample, key)
            if index is not None:
                new_value = new_value[index]
        else:                            
            new_value = converter(token)
        return new_value
        
        
if __name__ == '__main__':
    class DebugEventHandler:
        def __init__(self):
            pass
        
        def ack(self):
            print('ACK')
            
        def nack(self):
            print('NACK')
     
        
    class ResendingEventHandler:
        def __init__(self):
            self.resend = False
        
        def ack(self):
            self.resend = False
            
        def nack(self):
            print('NACK sent, resend required')
            self.resend = True      
        

    N = 20
    gen = VirtualLogGenerator()
    comms = VirtualCommChannel()
    event_handler = ResendingEventHandler()
    parser = LogParser(event_handler)
 
    samples = []
    while len(samples) < N:
        new_buffer = comms(gen.last() if event_handler.resend else gen.next())
        new_samples = parser(new_buffer)
        samples += new_samples
    
    for sample in samples:
        print(sample)
        print()
        
    print(parser.statistics)
        