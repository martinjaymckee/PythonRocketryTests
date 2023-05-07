def crc16_ccitt_false(data, crc=0xFFFF):
    t = 0
    for datum in data:
        t = (crc >> 8) ^ datum
        t = (t ^ (t >> 4))
        crc = (crc << 8) ^ (t << 12) ^ (t << 5) ^ t
        crc &= 0xFFFF
    return crc

class CRC16_CCITT_False:
    def __init__(self):
        self.__crc = None
        self.reset()
        
    @property
    def current(self):
        return (self.__crc & 0xFFFF)
    
    def reset(self):
        self.__crc = 0xFFFF
    
    def __call__(self, data):
        t = 0
        crc = self.__crc
        for datum in data:
            t = (crc >> 8) ^ datum
            t = (t ^ (t >> 4))
            crc = (crc << 8) ^ (t << 12) ^ (t << 5) ^ t
            crc &= 0xFFFF
        self.__crc = crc
        return crc
    
    