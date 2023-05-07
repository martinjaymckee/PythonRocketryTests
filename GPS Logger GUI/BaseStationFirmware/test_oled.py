import machine
import os
import random
import time

import sh1106

import timestamp


def gauss(mean, sd, N=12):
    t = 0
    for _ in range(N):
        t += random.uniform(-1, 1)
    return mean + (sd * t / float(N))

            
class HeaderInformation:
    def __init__(self):
        self.message = ''
        self.rf_status = None
        self.tracker_gps_status = None
        self.base_gps_status = None
        self.battery_voltage = 100


def radioFormat(status):
    if status in ['-', 'T']:
        return status
    return 'I'    


def gpsFormat(status):
    if status is None:
        return 'I'
    elif isinstance(status, int):
        if status < 0:
            return 'I'
        elif status <= 9:
            return str(status)
        else:
            return '*'
    elif status in ['-', 'T']:
        return status
    return 'I'


def writeHeader(display, info):
    width = 128
    characters = int(width / 8)
    display.fill_rect(0, 0, width, 10, 1)
    display.text(info.message, 0, 1, 0)
    
    if isinstance(info.rf_status, int):
        if info.rf_status > -50:
            pass
        if info.rf_status > -100:
            pass
        if info.rf_status > -150:
            pass
        display.hline(8*(characters-4) + 1, 8, 6, 0)
        display.fill_rect(8*(characters-4) + 1, 1, 2, 8, 0)
        display.fill_rect(8*(characters-4) + 3, 4, 2, 5, 0)
        display.fill_rect(8*(characters-4) + 5, 7, 2, 2, 0)        
    else:
        display.text(radioFormat(info.rf_status), 8*(characters-4), 1, 0)
        
    display.text(gpsFormat(info.tracker_gps_status), 8*(characters-3), 1, 0)
    display.text(gpsFormat(info.base_gps_status), 8*(characters-2), 1, 0)
    
    display.hline(8*(characters-1) + 2, 1, 4, 0)
    display.pixel(8*(characters-1) + 2, 2, 0)
    display.pixel(8*(characters-1) + 6, 2, 0)
    display.vline(8*(characters-1) + 1, 3, 5, 0)
    display.vline(8*(characters-1) + 6, 3, 5, 0)
    display.hline(8*(characters-1) + 1, 8, 6, 0)
    
#     display.text('B', 8*(characters-1), 1, 0)    
    
   
def df():
  s = os.statvfs('//')
  return ((s[0]*s[3])/1048576)


if __name__ == '__main__':
    dc_pin = machine.Pin(4)
    res_pin = machine.Pin(2)
    cs_pin = machine.Pin(5)
    spi = machine.SPI(2, baudrate=1000000)
    oled = sh1106.SH1106_SPI(128, 64, spi, dc_pin, res=res_pin, cs=cs_pin, rotate=180)
    header = HeaderInformation()
    header.message = 'Track To'
    header.rf_status = -116 #'-'
    header.tracker_gps_status = 6
    header.base_gps_status = 13
    
    t_last = timestamp.now()
    us = 1250*1000

    print('Free Memory: {} MB'.format(df()))
          
    while True:
        t = timestamp.now()
        expired, t_last = timestamp.expired_and_advance(t_last, us, t=t)
        if expired:
            oled.fill(0)
            header.tracker_gps_status = int(random.uniform(7, 9))
            header.base_gps_status = int(random.uniform(8, 13))
            writeHeader(oled, header)
            oled.text('Lat. {:0.4f}'.format(gauss(38.0, 25e-4)), 0, 11, 1)
            oled.text('Lon. {:0.4f}'.format(gauss(-105.12, 25e-4)), 0, 21, 1)
            oled.text('Alt. {:0.2f}'.format(gauss(2300, 1.5)), 0, 31, 1)                      
            oled.show()

