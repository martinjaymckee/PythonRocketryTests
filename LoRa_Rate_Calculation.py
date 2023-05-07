import math


class LoRaRateCalculator:
    __valid_bandwidths = [
        7800,
        10400,
        15600,
        20800,
        31200,
        41700,
        62500,
        125000,
        250000,
        500000
    ]

    __valid_sfs = [6, 7, 8, 9, 10, 11, 12]

    __valid_sfs = ['4/5', '4/6', '4/7', '4/8']

    def __init__(self, BW, SF, CR='4/5', suppress_exceptions=False):
        self.__BW = BW
        self.__SF = SF
        self.__CR = CR
        self.__suppress_exceptions = suppress_exceptions

    @property
    def symbol_rate(self):
        return self.__BW / pow(2, self.__SF)

    @property
    def symbol_time(self):
        return 1.0 / self.symbol_rate

    def preamble_time(self, preamble_symbols=8):
        return self.symbol_time * (4.25 + preamble_symbols)

    def payload_symbols(self, payload_bytes=0, packet_crc=True, explicit_header=False, de=False):
        payload_factor = 8 * payload_bytes
        sf_factor = 4 * self.__SF
        crc_factor = 16 if packet_crc else 0
        header_factor = 20 if explicit_header else 0
        # print('Payload Factor = {}'.format(payload_factor))
        # print('SF Factor = {}'.format(sf_factor))
        # print('CRC Factor = {}'.format(crc_factor))
        # print('Header Factor = {}'.format(header_factor))
        additional_payload = (payload_factor + 28 - sf_factor + crc_factor - header_factor)
        additional_payload /= (4*(self.__SF - (2 if de else 0)))
        additional_payload = (self.__cr_constant() + 4) * math.ceil(additional_payload)
        # print('additional_payload = {}'.format(additional_payload))
        return 8 + max(0, additional_payload)

    def payload_time(self, payload_bytes=0, packet_crc=True, explicit_header=False, de=False):
        n_payload = self.payload_symbols(payload_bytes, packet_crc, explicit_header, de)
        return n_payload * self.symbol_time

    def packet_time(self, payload_bytes=0, preamble_symbols=8, packet_crc=True, explicit_header=False, de=False):
        t_payload = self.payload_time(payload_bytes, packet_crc, explicit_header, de)
        t_preamble = self.preamble_time(preamble_symbols)
        return t_preamble + t_payload

    def payload_data_rate(self, payload_bytes=0, preamble_symbols=8, packet_crc=True, explicit_header=False, de=False):
        payload_bits = 8 * payload_bytes
        t_packet = self.packet_time(payload_bytes, preamble_symbols, packet_crc, explicit_header, de)
        f_max = 1 / t_packet
        return payload_bits * f_max

    def packet_duty_cycle(self, f_update, payload_bytes=0, preamble_symbols=8, packet_crc=True, explicit_header=False, de=False):
        t_packet = self.packet_time(payload_bytes, preamble_symbols, packet_crc, explicit_header, de)
        duty_cycle = t_packet / (1 / f_update)
        if not self.__suppress_exceptions:
            assert duty_cycle <= 1, 'Error: LoRa duty cycle exceeds 100% ({}%)'.format(int(100*duty_cycle))
        return duty_cycle

    def __cr_constant(self):
        _, _, a = self.__CR.partition('/')
        return int(a) - 4


def freespace_path_loss(f, d):
    c = 299792458
    return -20*math.log10(c/f) + 20*math.log10(d) + 20*math.log10(4*math.pi)


def two_ray_path_loss(f, d, ht, hr=None, Gr=1, Gt=1):
    # Note: Using approximated version from Wikipedia
    c = 299792458
    hr = ht if hr is None else hr
    lam = c / f
    rat = (4 * math.pi * ht * hr) / lam
    assert d > (10 * rat), 'Error: Distance between Rx and Tx insufficient'
    PL = (40 * math.log10(d) - (10 * math.log10(Gr * Gt * hr * ht)))
    return PL


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Freespace Path Loss of 435MHz signal over 60km = {} dB'.format(freespace_path_loss(435e6, 60e3)))
    print('Two Ray Path Loss of 435MHz signal over 2km at 1m above ground = {} dB'.format(two_ray_path_loss(435e6, 2e3, 2.5)))

    payload_bytes = list(range(16, 29))
    payload_bytes = list(range(15, 30))
    SFs = [6, 7, 8, 9, 10, 11, 12]

    explicit_header = True
    packet_crc = True
    duty_cycle_limit = 75

    f_fix = 4
    f_additional = 1
    BW = 125000
    CR = '4/8'

    fig, ax = plt.subplots(1, figsize=(16, 9), sharex=True, constrained_layout=True)

    f_update = f_fix + f_additional

    for SF in SFs:
        lora = LoRaRateCalculator(BW, SF, CR, suppress_exceptions=True)
        t_symbol = 1000 * lora.symbol_time
        # print('Symbol Time = {} ms\n'.format(1e3 * lora.symbol_time))
        # print('Packet duty cycle = {} %'.format(100 * lora.packet_duty_cycle(f_update, payload_bytes=payload_bytes, explicit_header=explicit_header, packet_crc=packet_crc)))
        duty_cycles = []
        valid = False
        for payload in payload_bytes:
            duty_cycle = 100 * lora.packet_duty_cycle(f_update, payload_bytes=payload, explicit_header=explicit_header, packet_crc=packet_crc)
            duty_cycle = None if duty_cycle > duty_cycle_limit else duty_cycle
            if duty_cycle is not None:
                valid = True
            duty_cycles.append(duty_cycle)
        if valid:
            ax.step(payload_bytes, duty_cycles, marker='x', where='post', label='SF = {}, Symbol Time = {} ms'.format(SF, t_symbol))
    ax.legend()
    # ax.set_yscale('log')
    ax.set_xlabel('Payload Bytes')
    ax.set_ylabel('TX Duty Cycle (at $f_{{update}} = {}$ Hz)'.format(f_update))
    plt.show()
