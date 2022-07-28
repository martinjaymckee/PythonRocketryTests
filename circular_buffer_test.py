import math
import struct


class OversampleBufferOptimized:
    def __init__(self, oversample_bits=6):
        self.__oversample_bits = oversample_bits
        self.__start = 0
        self.__end = 0
        self.__cnt = 0
        self.__capacity = int(4**oversample_bits)
        idx_bits = int(math.log2(self.__capacity)) + 1
        self.__idx_mask = (1 << idx_bits) - 1
        self.__buffer = bytearray([0]*(2*self.__capacity))
        self.__sum = 0
        self.__value = None

    @property
    def oversample_bits(self):
        return self.__oversample_bits

    @property
    def capacity(self):
        return self.__capacity

    def __len__(self):
        return self.__cnt

    def reset(self):
        self.__cnt = 0
        self.__start = 0
        self.__end = 0
        self.__sum = 0
        self.__value = None

    def __call__(self, v):
        if self.__cnt == self.capacity:
            end_idx = self.__end
            b = bytes([self.__buffer[end_idx], self.__buffer[end_idx + 1]])
            v_old = struct.unpack('h', b)[0]
            self.__sum -= v_old
            self.__end = (self.__end + 2) & self.__idx_mask
        b = struct.pack('h', v)
        start_idx = self.__start
        self.__buffer[start_idx] = b[0]
        self.__buffer[start_idx + 1] = b[1]
        self.__sum += v
        self.__start = (self.__start + 2) & self.__idx_mask
        if self.__cnt == self.capacity:
            self.__value = (self.__sum >> self.__oversample_bits)
        else:
            self.__cnt += 1
        return self.__value


class OversampleBuffer:
    def __init__(self, oversample_bits=6):
        self.__oversample_bits = oversample_bits
        self.__start = 0
        self.__end = 0
        self.__cnt = 0
        self.__capacity = int(4**oversample_bits)
        idx_bits = int(math.log2(self.__capacity))
        self.__idx_mask = (1 << idx_bits) - 1
        # print('capacity = {}, idx_bits = {}, idx_mask = 0x{:06X}'.format(self.__capacity, idx_bits, self.__idx_mask))
        self.__buffer = [0]*self.__capacity
        self.__sum = 0
        self.__value = None

    @property
    def oversample_bits(self):
        return self.__oversample_bits

    @property
    def capacity(self):
        return self.__capacity

    def __len__(self):
        return self.__cnt

    def reset(self):
        self.__cnt = 0
        self.__start = 0
        self.__end = 0
        self.__sum = 0
        self.__value = None

    def __call__(self, v):
        if self.__cnt == self.capacity:
            self.__sum -= self.__buffer[self.__end]
            self.__end = self.__wrap(self.__end + 1)
            # print('value = 0x{:04X}'.format(self.__value))
        self.__buffer[self.__start] = v
        self.__sum += v
        self.__start = self.__wrap(self.__start + 1)
        if self.__cnt == self.capacity:
            self.__value = (self.__sum >> self.__oversample_bits)
        else:
            self.__cnt += 1
        return self.__value

    def __wrap(self, idx):
        return idx & self.__idx_mask




if __name__ == '__main__':
    import time
    import random

    import matplotlib.pyplot as plt
    import seaborn as sns

    def getQuantizedValue(mean, sd, bits):
        max_value = (1<<bits) - 1
        return int(random.gauss(mean * max_value, sd * max_value) + 0.5)

    num_tests = 35
    min_oversample_bits = 2
    max_oversample_bits = 10
    bits = 12
    mean = 0.5
    sd = mean / 15
    oversamples = []
    errs = []
    max_num_samples = int(1.25*4**max_oversample_bits)
    in_max = (1 << bits) - 1

    t_test = 0
    num_passes = 0
    for test_idx in range(num_tests):
        print('Test {}'.format(test_idx))
        samples = [getQuantizedValue(mean, sd, bits) for _ in range(max_num_samples)]
        for oversample_bits in range(min_oversample_bits, max_oversample_bits+1):
            # buf = OversampleBufferOptimized(oversample_bits=oversample_bits)
            buf = OversampleBuffer(oversample_bits=oversample_bits)
            print('\tCreate Oversample Buffer with Capacity = {}'.format(buf.capacity))
            v = None
            max_val = (1 << (bits + buf.oversample_bits)) - 1
            tgt = getQuantizedValue(mean, 0, bits + buf.oversample_bits) / float(max_val)
            t_start = time.time()
            for v_update in samples:
                v = buf(v_update)
            dt = time.time() - t_start
            num_passes += 1
            if t_test is None:
                t_test = dt
            else:
                t_test += dt
            err = 100 * ((float(v) / max_val) - tgt) / tgt
            oversamples.append(oversample_bits)
            errs.append(err)

    # fig, ax = plt.subplots(1, constrained_layout=True)
    # samples = [getQuantizedValue(mean, sd, bits) for _ in range(10000)]
    # sns.histplot(samples, ax=ax)
    sns.jointplot(x=oversamples, y=errs, kind='hex')

    num_processed_samples = max_num_samples * num_passes
    print('Average Run Time Per Sample = {:0.4G} us'.format(1e6 * t_test / num_processed_samples))
    plt.show()
