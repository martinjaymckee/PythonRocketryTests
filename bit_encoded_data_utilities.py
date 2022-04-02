import math


def radius_equatorial():
    return 6378.1370e3 # WGS-84


def radius_polar():
    return 6356.7523e3 # WGS-84


def encoding_resolution(rng, bits):
    return (rng[1] - rng[0]) / (2**bits)


def encoding_bits(rng, res, update_range=False):
    bits = int(math.ceil(math.log2((rng[1] - rng[0]) / res)))
    return bits, rng


def calc_xyz_radius(h=0, rounding_digits=1):
    scale = 10**rounding_digits
    return scale * math.ceil((radius_equatorial() + h) / scale)


if __name__ == '__main__':
    #
    # Control Values
    #
    alt_max = 50e3
    vel_max = 3*343
    vel_min = 0.25
    l_error = 2.5
    res_scale = 75
    f_max = 5
    t_max = 5
    a_max = 16
    num_rel = f_max - 1
    header_bits = 21

    #
    # Calculated Values
    #
    g = 9.80665
    t_res = 0.01
    l_res_max = l_error / res_scale
    v_res_max = vel_min / res_scale
    lon_range = (-180, 180)
    lat_range = (-90, 90)
    xyz_radius = calc_xyz_radius(alt_max)
    xyz_range = (-xyz_radius, xyz_radius)
    alt_range = (-500, alt_max)
    vel_range = (-vel_max, vel_max)
    xyz_radius_rel = (vel_max / f_max)
    xyz_range_rel = (-xyz_radius_rel, xyz_radius_rel)
    vel_max_rel = (g * a_max) / f_max
    vel_range_rel = (-vel_max_rel, vel_max_rel)

    def deg_to_m(deg, surface_only=False, polar=False):
        r = radius_polar() if polar else radius_equatorial()
        c_total = (2 * math.pi * r) + (0 if surface_only else (2 * math.pi * alt_max))
        return (c_total / 360) * deg

    def m_to_deg(m, surface_only=False, polar=False):
        r = radius_polar() if polar else radius_equatorial()
        c_total = (2 * math.pi * r) + (0 if surface_only else (2 * math.pi * alt_max))
        return (360 * m) / c_total

    #
    # Best Encodings
    #
    lon_bits, lon_range = encoding_bits(lon_range, m_to_deg(l_res_max))
    lat_bits, lat_range = encoding_bits(lat_range, m_to_deg(l_res_max))
    alt_bits, alt_range = encoding_bits(alt_range, l_res_max)
    xyz_bits, xyz_range = encoding_bits(xyz_range, l_res_max)
    vel_bits, vel_range = encoding_bits(vel_range, v_res_max)
    xyz_rel_bits, xyz_rel_range = encoding_bits(xyz_range_rel, l_res_max)
    vel_rel_bits, vel_range_rel = encoding_bits(vel_range_rel, v_res_max)
    t_bits, t_range = encoding_bits((-0, t_max), t_res)

    print('Length Error = {} m'.format(l_error))
    print('Velocity Error = {} m/s'.format(vel_min))

    print('\nTarget length resolution = {} m'.format(l_res_max))
    print('Target Velocity resolution = {} m/s'.format(v_res_max))
    print()

    # lon_res_deg = encoding_resolution(lon_range, lon_bits)
    # lat_res_deg = encoding_resolution(lat_range, lat_bits)
    # print('Latitude/Longitude Resolution:')
    # print('\tLongitude Bits = {}'.format(lon_bits))
    # print('\tLongitude Degrees = {}'.format(lon_res_deg))
    # print('\tLongitude Meters = {}'.format(deg_to_m(lon_res_deg)))
    # print('\tLatitude Bits = {}'.format(lat_bits))
    # print('\tLatitude Degrees = {}'.format(lat_res_deg))
    # print('\tLatitude Meters = {}'.format(deg_to_m(lat_res_deg, polar=True)))
    # print()
    #
    # alt_res_m = encoding_resolution(alt_range, alt_bits)
    # print('Altitude Resolution:')
    # print('\tBits = {}'.format(alt_bits))
    # print('\tMaximum = {} km'.format(alt_max / 1000))
    # print('\tMeters = {}'.format(alt_res_m))
    # print()

    xyz_res_m = encoding_resolution(xyz_range, xyz_bits)
    print('Earth-Centered/Earth-Fixed Coordinate Resolution:')
    print('\tAbsolute:')
    print('\t\tBits = {}'.format(xyz_bits))
    print('\t\tRange = {}'.format(xyz_range))
    print('\t\tMeters = {}'.format(xyz_res_m))
    print('\tRelative:')
    print('\t\tBits = {}'.format(xyz_rel_bits))
    print('\t\tRange = {}'.format(xyz_range_rel))
    print()

    vel_res_mps = encoding_resolution(vel_range, vel_bits)
    print('Velocity Resolution:')
    print('\tAbsolute:')
    print('\t\tBits = {}'.format(vel_bits))
    print('\t\tRange = {}'.format(vel_range))
    print('\t\tMeters per Second = {}'.format(vel_res_mps))
    print('\tRelative:')
    print('\t\tBits = {}'.format(vel_rel_bits))
    print('\t\tRange = {}'.format(vel_range_rel))
    print()

    print('Time Resolution:')
    print('\tRelative:')
    print('\t\tBits = {}'.format(t_bits))
    print('\t\tRange = {}'.format(t_range))
    print()

    # print('Total LLH Storage Required = {} bits'.format(lon_bits + lat_bits + alt_bits + 3*vel_bits))
    ecef_abs_bits = 3*xyz_bits + 3*vel_bits
    ecef_rel_bits = num_rel * (3*xyz_rel_bits + 3*vel_rel_bits + t_bits)
    ecef_total_abs = (num_rel + 1) * (ecef_abs_bits + header_bits)
    ecef_total_comb = ecef_abs_bits + ecef_rel_bits + header_bits

    print('Total ECEF Storage Required (Absolute) = {} bits'.format(ecef_total_abs))
    print('Total ECEF Storage Required (Combination) = {} bits'.format(ecef_total_comb))
    print('\tBandwidth Reduction = {}%'.format(100 * (ecef_total_abs - ecef_total_comb) / float(ecef_total_abs)))
