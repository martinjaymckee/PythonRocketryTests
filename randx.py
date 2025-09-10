
import random

def gauss(mean, sd, N=12):
    t = 0
    for _ in range(N):
        t += random.uniform(-0.5, 0.5)
    return mean + (sd * t)