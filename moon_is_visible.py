import datetime
import math


class MoonAngleCalculator:
    def __init__(self, lat, lon):
        self.__new_moons = [
            datetime.datetime(2023, 10, 14, 17, 55),
            datetime.datetime(2023, 11, 13, 10, 50),
            datetime.datetime(2023, 12, 12, 23, 32),            
            datetime.datetime(2024, 1, 11, 11, 58),
            datetime.datetime(2024, 2, 9, 23, 0),
            datetime.datetime(2024, 3, 10, 9, 2),
            datetime.datetime(2024, 4, 8, 18, 23),
            datetime.datetime(2024, 5, 7, 3, 24),
            datetime.datetime(2024, 6, 6, 13, 40)
        ]
        self.__new_moons.sort()
        self.__lat = lat
        self.__lon = lon

    def __call__(self, td):
        nearest_new_moon, diff_td, cycle_time = self.__get_new_moon_parameters(td)
        moon_angle = 2 * math.pi * diff_td.total_seconds() / cycle_time.total_seconds()
        print('nearest = {}, diff_td = {}, cycle_time = {}, moon_angle = {}'.format(nearest_new_moon, diff_td, cycle_time, 57.3 * moon_angle))
        return moon_angle
        
    def __get_new_moon_parameters(self, td):
        last_idx = None
        for idx, new_moon_td in enumerate(self.__new_moons):
            if td > new_moon_td:
                last_idx = idx
                break
        last_td, next_td = None, None
        if (last_idx is None) or (last_idx >= len(self.__new_moons) - 1):
            last_td = self.__new_moons[-2]
            next_td = self.__new_moons[-1]
        else:
            last_td = self.__new_moons[last_idx]
            next_td = self.__new_moons[last_idx + 1]
        cycle_time = next_td - last_td
        diff_last = (td-last_td) if (td > last_td) else (last_td-td)
        diff_next = (td-next_td) if (td > next_td) else (next_td-td)
        nearest_td = None
        diff_td = None
        if diff_last < diff_next:
            nearest_td, diff_td = last_td, diff_last
        else:
            nearest_td, diff_td = next_td, cycle_time - diff_next
        return nearest_td, diff_td, cycle_time


class TimeAngleCalculator:
    def __init__(self, lat, lon):
        self.__day_duration = float(60 * 60 * 24)
        self.__lat = lat
        self.__lon = lon

    def __call__(self, td):
        #t = td.time()
        noon = datetime.datetime(td.year, td.month, td.day, 12, 0, 0)#, tzinfo=datetime.timezone.utc)
        print('td = {}, noon = {}, diff = {}'.format(td, noon, (td-noon).total_seconds()))
        time_angle = math.pi * (td - noon).total_seconds() / self.__day_duration
        print('time_angle = {}'.format(time_angle))
        print('lon = {}'.format(self.__lon))
   #     time_angle += self.__lon
        return time_angle
        

class MoonVisibilityCalculator:
    def __init__(self, lat, lon):
        self.__lat = math.radians(lat)
        self.__lon = math.radians(lon)
        self.__view_half_angle = math.radians(90.83)
        self.__moon_angle_calculator = MoonAngleCalculator(self.__lat, self.__lon)
        self.__time_angle_calculator = TimeAngleCalculator(self.__lat, self.__lon)
        self.__debug = False

    def __call__(self, td):
        #td = td.astimezone(tz=None)
        moon_angle = self.__moon_angle_calculator(td)
        time_angle = self.__time_angle_calculator(td)
        centered_moon_angle = -(2*math.pi - moon_angle) if moon_angle > math.pi else moon_angle
        angle_diff = centered_moon_angle - time_angle
        print('centered_moon_angle = {}, time_angle = {}, diff = {}'.format(57.3*centered_moon_angle, 57.3*time_angle, 57.3*angle_diff))
        visibility = abs(angle_diff) < self.__view_half_angle

        if self.__debug:
            import matplotlib.pyplot as plt
            r_sun, r_earth, r_moon = 2, 1, 0.33
            l_earth_moon, r_moon_orbit = 20, 8
            fig, ax = plt.subplots(1, constrained_layout=True)
            ax.set_aspect(1)
            ax.add_artist(plt.Circle((0, 0), r_earth, color='b'))
            ax.add_artist(plt.Circle((20, 0), r_sun, color='y'))
            x_moon, y_moon = r_moon_orbit*math.cos(moon_angle), r_moon_orbit*math.sin(moon_angle)
            ax.add_artist(plt.Circle((x_moon, y_moon), r_moon, color='k', fill=False))
            x_time, y_time = r_earth*math.cos(time_angle), r_earth*math.sin(time_angle)
            for offset in [-math.pi / 2, math.pi / 2]:
                x_time_end = x_time + (2 * l_earth_moon * math.cos(time_angle + offset))
                y_time_end = y_time + (2 * l_earth_moon * math.sin(time_angle + offset))
                ax.add_artist(plt.Line2D([x_time, x_time_end], [y_time, y_time_end], color='c'))
            ax.scatter([x_time], [y_time], c='m')
            ax.set_xlim(-r_moon_orbit+(2*r_moon), l_earth_moon+(2*r_sun))
            ax.set_ylim(-r_moon_orbit+(2*r_moon), r_moon_orbit+(2*r_moon))
            plt.show()
        return visibility


def is_leap_year_gregorian(Y):
    div = 4 if (not (Y % 100) == 0) else 16
    return (Y & (div-1)) == 0


def fixed_from_gregorian(Y, M, d):
    Y0 = Y - 1
    result = (365 * Y0) + int(math.floor(Y0 / 4)) - int(math.floor(Y0 / 100)) + int(math.floor(Y0 / 400))
    result += int(math.ceil(((367*M) - 362) / 12)) + d
    return result - (0 if M <= 2 else (1 if is_leap_year_gregorian(Y) else 2))


def day_number_gregorian(Y, M, d):
    return fixed_from_gregorian(Y, M, d) - fixed_from_gregorian(Y-1, 12, 31)


class SolarPositionCalculator:
    def __init__(self, lat, lon, tz_offset=-7):
        self.__lat = math.radians(lat)
        self.__lon = lon
        self.__tz_offset = tz_offset

    def sunrise_and_sunset(self, date, zenith=90.83):
        Y, M, d, h, m = date.year, date.month, date.day, date.hour, date.minute         
        gamma = self.__fractional_year(Y, M, d, h, m)
        eqtime = self.__eqtime(gamma)
        ha = self.__hour_angle(gamma, zenith)
        sunrise_minutes = 720 - (4 * (self.__lon + ha)) - eqtime
        sunset_minutes = 720 - (4 * (self.__lon - ha)) - eqtime
        sunrise = self.__min_to_time(sunrise_minutes)
        sunset = self.__min_to_time(sunset_minutes)
        return sunrise, sunset
        #return sunrise_minutes, sunset_minutes

    def __hour_angle(self, gamma, zenith=90.83):
        zenith = math.radians(zenith)
        decl = self.__decl(gamma)
        return math.degrees(math.acos((math.cos(zenith) / (math.cos(self.__lat) * math.cos(decl))) - (math.tan(self.__lat) * math.tan(decl))))

    def __fractional_year(self, Y, M, d, h, m):
        # NOTE: CURRENTLY THE MINUTES ARE NOT BEING USED TO CALCULATE THE FRACTIONAL DAY
        yd = day_number_gregorian(Y, M, d)
        days = 366 if is_leap_year_gregorian(Y) else 365
        return ((2 * math.pi) / days) * ((yd - 1) + ((h - 12) / 24))
    
    def __eqtime(self, gamma):
        gamma = math.radians(gamma)
        return 229.18 * (0.000075 + (0.001868 * math.cos(gamma)) - (0.032077 * math.sin(gamma)) - (0.01461 * math.cos(2*gamma)) - (0.040849 * math.sin(2 * gamma)))

    def __decl(self, gamma):
        gamma = math.radians(gamma)
        return 0.006918 - (0.399912 * math.cos(gamma)) + (0.070257 * math.sin(gamma)) - (0.006758 * math.cos(2 * gamma)) + (0.000907 * math.sin(2 * gamma)) - (0.002697 * math.cos(3 * gamma)) + (0.00148 * math.sin(3 * gamma))

    def __min_to_time(self, m):
        m += (60 * self.__tz_offset)
        h = int(m / 60) 
        m -= (60 * h)
        s = int(60*(m - int(m)))
        m = int(m)
        if h < 0:
            h += 24 # TODO: CHECK IF THIS IS CORRECT
        return (h, m, s)


if __name__ == '__main__':
    def is_leap_year(Y):
        return is_leap_year_gregorian(Y)

    print('is_leap_year(1900) = {}'.format(is_leap_year(1900)))
    print('is_leap_year(1980) = {}'.format(is_leap_year(1980)))
    print('is_leap_year(1981) = {}'.format(is_leap_year(1981)))
    print('is_leap_year(1982) = {}'.format(is_leap_year(1982)))
    print('is_leap_year(1983) = {}'.format(is_leap_year(1983)))
    print('is_leap_year(1984) = {}'.format(is_leap_year(1984)))   
    print('is_leap_year(2000) = {}'.format(is_leap_year(2000)))
    print('is_leap_year(2001) = {}'.format(is_leap_year(2001)))    

    current_date = datetime.datetime.now()
    print('day_number_gregorian(today) = {}'.format(day_number_gregorian(current_date.year, current_date.month, current_date.day)))

    lat, lon = 38.8339, -104.8214
    date = datetime.datetime.now()
    moon_visibility_calculator = MoonVisibilityCalculator(lat, lon)
    visibility = moon_visibility_calculator(date)
    print('moon_is_visible = {}'.format(visibility))

    solar_position_calculator = SolarPositionCalculator(lat, lon)
    sunrise, sunset = solar_position_calculator.sunrise_and_sunset(date)
    #print('sunrise_minutes = {}, sunset_minutes = {}'.format(sunrise, sunset))    
    print('sunrise = {:02d}:{:02d}:{:02d}, sunset = {:02d}:{:02d}:{:02d}'.format(*sunrise, *sunset))