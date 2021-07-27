import functools
import math
import random

import numpy as np


@functools.total_ordering
class NormalRandomVariable:
    @classmethod
    def Construct(cls, val):
        """
        This method will either create a copy of the random variable or, if the passed
        value is a scalar, create a new random variable with zero variance.
        """
        if isinstance(val, cls):
            return cls(mean=val.mean, variance=val.variance)
        return cls(mean=val)

    @classmethod
    def Noise(cls, sd=0, variance=None):
        return cls(mean=0, variance=sd**2 if variance is None else variance)

    def __init__(self, mean, sd=0, variance=None):
        self.__mean = mean
        self.__variance = (sd**2) if variance is None else variance

    def __eq__(self, other):
        return self.mean == mean(other)

    def __lt__(self, other):
        return self.mean < mean(other)

    @property
    def mean(self):
        return self.__mean

    @property
    def variance(self):
        return self.__variance

    @property
    def standard_deviation(self):
        return math.sqrt(self.__variance)

    def sample(self, N, dtype=float):
        samples = [dtype(self) for _ in range(N)]
        return np.array(samples)

    def __str__(self):
        return 'N({:g}, {:g})'.format(self.__mean, self.__variance)

    def __float__(self):
        return float(random.gauss(self.mean, self.standard_deviation))

    def __int__(self):
        return int(random.gauss(self.mean, self.standard_deviation))

    def __neg__(self):
        return self.__class__(mean=-self.__mean, variance=self.__variance)

    def __add__(self, other):
        other = self.__class__.Construct(other)
        new_mean = self.__mean + other.mean
        new_variance = self.__variance + other.variance
        return self.__class__(mean=new_mean, variance=new_variance)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        other = self.__class__.Construct(other)
        return other - self

    def __mul__(self, other):
        other = self.__class__.Construct(other)
        mx, my = self.__mean, other.mean
        vx, vy = self.__variance, other.variance
        new_mean = mx * my
        # print('mx = {}, my = {}, vx = {}, vy = {}'.format(mx, my, vx, vy))
        new_variance = ((vx + mx**2)*(vy + my**2)) - (mx**2 * my**2)
        return self.__class__(mean=new_mean, variance=new_variance)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self.__class__.Construct(other)
        mx, my = self.__mean, other.mean
        vx, vy = self.__variance, other.variance
        new_mean = mx / my
        new_variance = (mx**2/my**2)*((vx/mx**2) + (vy/my**2))
        return self.__class__(mean=new_mean, variance=new_variance)

    def __rtruediv__(self, other):
        other = self.__class__.Construct(other)
        return other.__truediv__(self)


NRV = NormalRandomVariable


def mean(val):
    if isinstance(val, NormalRandomVariable):
        return val.mean
    return val


def variance(val):
    if isinstance(val, NormalRandomVariable):
        return val.variance
    return 0


def standard_deviation(val):
    if isinstance(val, NormalRandomVariable):
        return val.standard_deviation
    return 0


def abs(val):
    if isinstance(val, NormalRandomVariable):
        val = val.mean
    return abs(val)


def oversample(val, samples):
    if isinstance(val, NormalRandomVariable):
        return NRV(val.mean, val.standard_deviation / math.sqrt(samples))
    return val


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    vx, vy = 0.1, 0.15
    one = NRV(mean=1, variance=vx)
    half = NRV(mean=0.5, variance=vy)
    print('-N(1, {:g}) = {}'.format(vx, -one))
    print('N(1, {:g}) + N(0.5, {:g}) = {}'.format(vx, vy, one + half))
    print('N(1, {:g}) - N(0.5, {:g}) = {}'.format(vx, vy, one - half))
    print('1 + N(0.5, {:g}) = {}'.format(vy, 1 + half))
    print('N(0.5, {:g}) + 1 = {}'.format(vy, half + 1))
    print('1 - N(0.5, {:g}) = {}'.format(vy, 1 - half))
    print('N(0.5, {:g}) - 1 = {}'.format(vy, half - 1))
    print('N(1, {:g}) * N(0.5, {:g}) = {}'.format(vx, vy, one * half))
    print('N(0.5, {:g}) * N(1, {:g}) = {}'.format(vy, vx, half * one))
    print('1 * N(0.5, {:g}) = {}'.format(vy, 1 * half))
    print('N(0.5, {:g}) * 1 = {}'.format(vy, half * 1))
    print('N(1, {:g}) / N(0.5, {:g}) = {}'.format(vx, vy, one / half))
    print('N(0.5, {:g}) / N(1, {:g}) = {}'.format(vy, vx, half / one))
    print('1 * N(0.5, {:g}) = {}'.format(vy, 1 / half))
    print('N(0.5, {:g}) / 1 = {}'.format(vy, half / 1))

    print('N(1, 0.5) / 10 = {}'.format(NRV(1, 0.5) / 10))
    print('10 * N(1, 0.5) = {}'.format(10 * NRV(1, 0.5)))

    onehundred = NRV(mean=100, variance=100)
    samples = onehundred.sample(10000, dtype=int)

    sns.distplot(samples)
    plt.show()
