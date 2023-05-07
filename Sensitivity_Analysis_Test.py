# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 07:33:11 2022

@author: marti
"""

import math
import random


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg


class Igniter:
    def __init__(self, Rs, Ifire):
        self.Rs = Rs
        self.Ifire = Ifire
        
    def __str__(self):
        return 'Igniter( Rs = {:0.2f} ohms, Ifire = {:d} mA )'.format(self.Rs, int(1000 * self.Ifire))
        
    
class IgniterPopulation:
    def __init__(self, Rs, Ifire, Ifire_sd, Rs_sd = 0):
        self.__Rs = Rs
        self.__Rs_sd = Rs_sd
        self.__Ifire = Ifire
        self.__Ifire_sd = Ifire_sd
        self.__N = 3 # NOTE: THIS IS HOW MANY STANDARD DEVIATIONS TO GO FOR NFC AND AFC

    def __str__(self):
        return 'IgniterPopulation( Mean Rs = {:0.2f} ohms, NFC = {:d} mA, AFC = {:d} mA )'.format(self.__Rs, int(1000 * self.no_fire_current), int(1000 * self.all_fire_current))
    
    @property
    def threshold_current(self):
        return self.__Ifire
    
    @property
    def all_fire_current(self):
        return self.__Ifire + (self.__N * self.__Ifire_sd)

    @property
    def no_fire_current(self):
        return self.__Ifire - (self.__N * self.__Ifire_sd)

    def sample(self):
        Rs = random.gauss(self.__Rs, max(self.__Rs_sd, 1e-9))
        Ifire = random.gauss(self.__Ifire, self.__Ifire_sd)
        return Igniter(Rs, Ifire)
        
        
def igniterFires(Ifire, ign):
    return Ifire >= ign.Ifire


class CountEndTrigger:
    def __init__(self, count):
        self.__end_count = count
        
    def __call__(self, count, successes, failures, **kwargs):
        return count >= self.__end_count
        
        
class BrucetonMethod:
    def __init__(self, drive, step):
        self.__drive = drive
        self.__step = step

    @property
    def drive(self):
        return self.__drive
        
    def next(self, was_success):
        change = -self.__step if was_success else self.__step
        self.__drive += change
        return self.__drive, change
    

class RobbinsMonroMethod:
    def __init__(self, drive, step, scaling=0.9):
        self.__drive = drive
        self.__step = step
        self.__last_success = None
        self.__scaling = scaling
        
    @property
    def drive(self):
        return self.__drive
        
    def next(self, was_success):
        if self.__last_success is None:
            self.__last_success = was_success
        if not self.__last_success == was_success:
            self.__step *= self.__scaling
        change = -self.__step if was_success else self.__step
        self.__drive += change
        return self.__drive, change
    
    
def runSensitivityAnalysis(ign_pop, algo, end_trigger=50, voltage_drive=True):
    if isinstance(end_trigger, int):
        end_trigger = CountEndTrigger(end_trigger)
        
    count = 0
    Itest = None
    num_successes = 0
    num_failures = 0
    end_trigger_kwargs = {}
    drive = algo.drive

    data = []
    
    while not end_trigger(count, num_successes, num_failures, **end_trigger_kwargs):
 # TODO: ADD ERROR CHECKING FOR NEGATIVE DRIVE LEVELS... OTHER ERRORS?
        count = count + 1
        ign = ign_pop.sample()
        Itest = drive / ign.Rs if voltage_drive else drive
        success = igniterFires(Itest, ign)
        if success:
            num_successes = num_successes + 1
        else:
            num_failures = num_failures + 1
        data.append( (count, num_successes, num_failures, ign, drive, Itest, success) )
        drive, _ = algo.next(success)
        
    return data
        


class SensitivityAnalysisStatistics:
    def __init__(self, learning_rate=0.025, threshold=0.5):
        self.__learning_rate = learning_rate
        self.__threshold = threshold

    def __call__(self, data, M=50):
        N = len(data)
        print('N = {}'.format(N))
        
#        mu = 0.5
#        s = 0.025

        mu = 1
        s = 0.1
        
        # Create Base Variables
        W = np.zeros((N, N))
        X = np.ones((N, 2))
        Y = np.zeros((N, 1))
        Y_hat = np.zeros((N, 1))
        
        minimum_success = None
        maximum_failure = None
        for idx, test in enumerate(data):
            Itest, success = test[5], test[6]
            X[idx, 1] = Itest
            Y[idx] = 1.0 if success else 0.0
            if success: # Check Success current
                if minimum_success is None or minimum_success > Itest:
                    minimum_success = Itest
            else: # Check Failure Current
                if maximum_failure is None or maximum_failure < Itest:
                    maximum_failure = Itest
        beta = np.array([-mu/s, 1/s])
        categories_overlap = (minimum_success is not None) and (maximum_failure is not None) and minimum_success < maximum_failure

        if not categories_overlap:
            print('No unique solution to logistic regression')
            return None
        
        # Do Iterations of Logistic Regression
        for iteration in range(M):
            # Create W matrix
            for (count, _, _, _, _, Itest, success) in data:
                idx = count - 1
                success = 1.0 if success else 0.0
                p = self.p(Itest, beta)
                W[idx][idx] = p * (1-p)
                Y_hat[idx] = 1.0 if p > self.__threshold else 0.0
            dbeta = self.__learning_rate * np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(X.T, W),X)), X.T), (Y - Y_hat))
            beta[0] += dbeta[0, 0]
            beta[1] += dbeta[1, 0]
            #print(W)
        print('beta = {}'.format(beta))
        print('mu = {}'.format(-beta[0]/beta[1]))
        print('s = {}'.format(1/beta[1]))
        print('Q: 10% = {}, 90% = {}'.format(self.q(0.1, beta), self.q(0.9, beta)))
        print('Q: 5% = {}, 95% = {}'.format(self.q(0.05, beta), self.q(0.95, beta)))
        print('Q: 1% = {}, 99% = {}'.format(self.q(0.01, beta), self.q(0.99, beta)))
        
        return beta
    
    def p(self, I, beta):
        return 1 / (1 + math.exp(-(beta[1]*I + beta[0])))
    
    def y_hat(self, I, beta):
        p = self.p(I, beta)
        return 1.0 if p > self.__threshold else 0.0
    
    def q(self, p, beta):
        mu = -beta[0] / beta[1]
        s = 1 / beta[1]
        return mu + s * math.log(p / (1 - p))
        
    
    
if __name__ == '__main__':
    N = 100
    population = IgniterPopulation(1.2, 0.3, 0.005, Rs_sd=0.25)
    print(population)
    
#    for _ in range(N):
#        ign = population.sample()
#        print(ign)
#    algo = BrucetonMethod(0.2, 0.004)
    algo = RobbinsMonroMethod(0, 0.01, 0.75)
    
    data = runSensitivityAnalysis(population, algo, end_trigger=N)

    processor = SensitivityAnalysisStatistics()
    beta = processor(data)
    

    counts = []
    Is = []
    results = []
    result_colors = []
    predicted = []
    predicted_colors = []
    
    for count, num_successes, num_failures, ign, drive, Itest, success in data:
        counts.append(count)
        Is.append(Itest)
        results.append(int(success))
        result_colors.append('g' if success else 'r')
        predicted.append(processor.y_hat(Itest, beta))
        predicted_colors.append('c' if success else 'm')
 
    fig, axs = plt.subplots(2, constrained_layout=True)
#    axs[0].scatter(counts, predicted, c=predicted_colors)
    axs[0].scatter(counts, results, c=result_colors)
    ax0 = axs[0].twinx()
    ax0.scatter(counts, Is, c='c', alpha=0.25)
    axs[1].scatter(Is, results, c='g')
    axs[1].scatter(Is, predicted, c='c', alpha=0.5)
    axs[1].axvline(population.no_fire_current)
    axs[1].axvline(population.threshold_current)
    axs[1].axvline(population.all_fire_current)
 
    ax1 = axs[1].twinx()
    Is_sorted = sorted(Is)
    ps = [processor.p(I, beta) for I in Is_sorted]
    ax1.plot(Is_sorted, ps, c = 'k', alpha=0.5)
    plt.show()
        