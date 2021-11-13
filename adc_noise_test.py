#
# Standard Library
#
import os
import os.path
import signal
# import sys
import time


#
# Import 3rd Party Libraries
#
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import serial
import serial.tools.list_ports


class ADCNoiseParser:
    def __init__(self):
        self.__buffer = ''

    def parse(self, data):
        self.__buffer += data.decode('ascii', 'ignore')
        samples = []
        while self.__buffer.find('\n') >= 0:
            beginning, _, remaining = self.__buffer.partition('\n')
            self.__buffer = remaining
            beginning = beginning.strip()
            if beginning.startswith('$'):
                tokens = beginning[1:].split(',')
                if len(tokens) == 3:
                    try:
                        t, val, crc = [int(v) for v in tokens]
                        if crc == ((t + val) & 0xFFFFFFFF):
                            samples.append((t, val))
                    except Exception as e:
                        pass
        # if len(samples) > 0:
        #     print('{} new samples'.format(len(samples)))
        return samples

    def dump_buffer(self):
        print(self.__buffer)


class ADCReferenceValues:
    def __init__(self, per_reference=500):
        self.__per_reference = per_reference
        self.__v_refs = []
        self.__next_ref_count = 0

    @property
    def reference_points(self):
        return self.__v_refs[:]

    def update(self, samples, force=False):
        num_samples = len(samples)
        if (num_samples >= self.__next_ref_count) or force:
            v_in = input('Current measured voltage? ')
            v_ref = float(v_in)
            self.__v_refs.append((num_samples, v_ref))
            self.__next_ref_count += self.__per_reference


def save_data(filename, ts, vs):
    print('Save Data to {}'.format(filename))
    path = os.path.join('data', filename)
    with open(path, 'w') as file:
        for t, v in zip(ts, vs):
            file.write('{}, {}\r\n'.format(t, v))


if __name__ == '__main__':
    running = True
    for info in serial.tools.list_ports.comports():
        print(info)

    port = 'COM5'
    baud = 115200
    parser = ADCNoiseParser()
    ref_processor = ADCReferenceValues()
    output_directory = 'data'
    input_filename = 'adc_noise_test_input_21.txt'
    data_filename = 'adc_noise_test_data_21.csv'

    input_buffer = ''

    try:
        print('Connecting to Serial Port - {}'.format(port))
        comm = serial.Serial(port, baudrate=baud)
    except Exception as e:
        print(e)
        quit(-1)

    def signal_handler(*args):
        global running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    samples = []
    v_refs = []
    required_samples = 25000
    count = 0
    try:
        # ref_processor.update(samples)
        while running:
            if comm.in_waiting > 0:
                new_data = comm.read(comm.in_waiting)
                input_buffer += new_data.decode('ascii', 'ignore')
                new_samples = parser.parse(new_data)
                samples += new_samples
                count += len(new_samples)
            if count > 500:
                print('.', end='', flush=True)
                count -= 500
            if len(samples) >= required_samples:
                running = False
    finally:
        while comm.out_waiting > 0:
            time.sleep(0.01)
        comm.close()
        print('Closed Comm Port')

    # ref_processor.update(samples, force=True)
    print('Number of samples = {}'.format(len(samples)))
    ts = np.array([t for t, _ in samples])
    vs = np.array([v for _, v in samples])
    print('Mean value = {}'.format(np.mean(vs)))

    save_data(data_filename, ts, vs)

    print('Save input buffer to {}'.format(input_filename))
    with open(os.path.join('data', input_filename), 'w') as file:
        file.write(input_buffer)

    # print(ref_processor.reference_points)
    # fig, ax = plt.subplots(2, figsize=(16, 9), constrained_layout=True)
    # sns.distplot(vs, ax=ax)
    # sns.jointplot(x=ts, y=vs, kind='reg')
    # plt.show()
