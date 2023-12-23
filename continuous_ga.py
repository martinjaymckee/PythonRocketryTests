import math
import numbers
import random

import numpy as np

class ContinuousChromosome(list):
    def __init__(self, yp, *args, **kwargs):
        super().__init__(yp)
        self.__args = args
        self.__kwargs = kwargs
        
    @property
    def yp(self):
        return self
    
    def __call__(self, *args, **kwargs):
        assert False, 'Error: {}.__call__() is undefined!'.format(self.__class__.__name__)

    def mutate_with_p(self, p, sd=None):
        N = len(self)
        for idx in range(N):
            if p < random.uniform(0, 1):
                sd2 = abs(self[idx]) / 10 if sd is None else sd
                self[idx] = random.gauss(self[idx], sd2)
        return True
    
    def mutate_all(self, sd=None):
        return mutate_with_p(1.0, sd)

    def clone_with(self, yp):
        # print('yp = {}'.format(yp))
        return self.__class__(yp, self.__args, self.__kwargs)


def crossover(individual_a, individual_b, first_only=False):
    assert type(individual_a) == type(individual_b), 'Error in crossover operation, {} is not the same type as {}'.format(type(individual_a), type(individual_b))

    def do_crossover(idx, a, b):
        first = list(a[:idx])
        second = list(b[idx:])
        result = first + second
        # print('first = {}, second = {}, result = {}'.format(first, second, result))
        return a.clone_with(result)
    
    p = random.uniform(0, 1)
    idx = int(math.floor((min(len(individual_a), len(individual_b)) + 1) * p))
    # print('p = {}, idx = {}'.format(p, idx))
    children = []
    
    
    children.append(do_crossover(idx, individual_a, individual_b))
    
    if not first_only:
        children.append(do_crossover(idx, individual_b, individual_a))
        
    return children
    

class EiffelTowerChromosome:
    def __init__(self, Re_max, ps=[], cd_min=0.05, cd_max=5.0):
        self.__Re_max = float(Re_max)
        ps = [float(p) for p in ps]
        self.__xp = np.linspace(0, Re_max, len(ps))
        self.__yp = np.array(ps)
        self.__interp = scipy.interpolate.interp1d(self.__xp, self.__yp)
        self.__cd_min = cd_min
        self.__cd_max = cd_max
        
    @property
    def xp(self):
        return self.__xp
    
    @property
    def yp(self):
        return self.__yp
        
    def __call__(self, Re):
        if Re < 0:
            raise Exception('Error in EiffelTowerChromosome.__call__: Negative Reynolds Number passed in!')
            
        if Re >= self.__Re_max: # Saturate beyond maximum Re
            return self.__yp[-1]
        
        return self.__interp(Re)


class GeneticAlgorithmConfiguration:
    default_elitism = 0.175
    default_max_generations = 1000
    default_mutation_rate = 0.2
    
    def __init__(self, elitism=None, mutation_rate=None, max_generations=None):
        self.__elitism = GeneticAlgorithmConfiguration.default_elitism if elitism is None else elitism
        self.__max_generations = GeneticAlgorithmConfiguration.default_max_generations if max_generations is None else max_generations
        self.__mutation_rate = GeneticAlgorithmConfiguration.default_mutation_rate if mutation_rate is None else mutation_rate
        
    @property
    def elitism(self):
        return self.__elitism
    
    @property
    def max_generations(self):
        return self.__max_generations
    
    @property
    def mutation_rate(self):
        return self.__mutation_rate
    
    
class GeneticAlgorithmStatistics:
    def __init__(self, generation, pop_size, num_elite, best_score, best_individual, mean_score):
        self.generation = generation
        self.num_elite = num_elite
        self.best_score = best_score
        self.best_individual = best_individual
        self.mean_score = mean_score
        self.pop_size = pop_size
    
    
class ContinuousGeneticAlgorithm:
    def __init__(self, population, fitness, ending=None, config=None, callbacks={}):
        self.__population = population
        self.__fitness = fitness
        self.__ending = ending
        self.__config = GeneticAlgorithmConfiguration() if config is None else config
        self.__callbacks = callbacks

    def run(self, max_generations=None, sd=None):
        statistics = []
        generation = 0
        pop_size = len(self.__population)
        num_elite = 0 if self.__config is None else int(self.__config.elitism * pop_size)
        
#        print('pop_size = {}, num_elite = {}'.format(pop_size, num_elite))
#        print('elitism = {}'.format(self.__config.elitism))
        if max_generations is None:
            if self.__config is not None:
                max_generations = self.__config.max_generations
            else:
                max_generations = GeneticAlgorithmConfiguration.default_max_generations
              
        while (not self.__ending_criteria_met(statistics)) and (generation < max_generations): # TODO: ADD ENDING CRITERIA
            p_mutate = GeneticAlgorithmConfiguration.default_mutation_rate
            if self.__config is not None:
                if isinstance(self.__config.mutation_rate, numbers.Number):
                    p_mutate = self.__config.mutation_rate
                else:
                    p_mutate = self.__config.mutation_rate(generation) # TODO: DECIDE IF THIS IS THE ONLY ARGUMENT
#            print('Generation {}, p_mutate = {}'.format(generation, p_mutate))
            if 'begin_generation' in self.__callbacks:
                self.__callbacks['begin_generation'](generation)
            scores = []
            for c in self.__population:
                fit = self.__fitness(c)
                scores.append( (fit, c) )
                if 'calc_individual' in self.__callbacks:
                    self.__callbacks['calc_individual'](c)                
            scores.sort(key = lambda x: x[0])
            # for score, c in scores: # TODO: ADD A CALLBACK TO PROCESS THE SCORES???
            #     c_fmt = ','.join(['{:0.4f}'.format(v) for v in c])
            #     print('\t{:0.4}: [{}]'.format(score, c_fmt))
                
            children = [scores[idx][1] for idx in range(num_elite)]
#            print('\tElite Children = {}'.format(children))
            score_values = np.array([score[0] for score in scores])
            score_min, score_max = score_values[0], score_values[-1]
            best_individual = scores[0][1]
            pass_stats = GeneticAlgorithmStatistics(generation, pop_size, num_elite, score_min, best_individual, np.mean(score_values))
            statistics.append(pass_stats)
            w = list( (score_max - score_values) / (score_max - score_min) ) # TODO: THIS IS CURRENTLY NOT WORKING CORRECTLY AS IT WILL ALWAYS IGNORE THE WORST SCORE... THERE SHOULD BE A BETTER WAY
            #print(w)
            #print(score_values)
            while len(children) < pop_size:
                individual_a, individual_b = random.choices(self.__population, w, k=2)
                new_children = crossover(individual_a, individual_b)
                for child in new_children:
                    child.mutate_with_p(p_mutate, sd=sd)
                    children.append(child)
                    if len(children) == pop_size:
                        break
                #print(individual_a, individual_b)
            self.__population = children
            if 'end_generation' in self.__callbacks:
                self.__callbacks['end_generation'](generation, pass_stats)             
            generation = generation + 1           
        return statistics

    def __ending_criteria_met(self, stats):
        if (self.__ending is None) or (stats is None):
            return False
        try:
            return self.__ending(stats)
        except Exception:
            pass
        return False
    
    
def plot_ga_scores(stats):
    generations = np.array([stat.generation for stat in stats])    
    best_scores = np.array([stat.best_score for stat in stats])
    mean_scores = np.array([stat.mean_score for stat in stats])            

    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title('Mean Scores (blue), Best Scores (green)')
    ax.plot(generations, mean_scores, c='b')
    ax0 = ax.twinx()
    ax0.plot(generations, best_scores, c='g')
    ax0.yaxis.label.set_color('g')
    ax.tick_params(axis='y', labelcolor='b')
    ax0.tick_params(axis='y', labelcolor='g')

    return fig, [ax, ax0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    target_chromosome = [2.5, -7.6, 0.15]
    def generate_population(N=25):
        population = []
        for _ in range(N):
            vals = [random.uniform(-10, 10) for _ in range(3)]
            population.append(ContinuousChromosome(vals))
        return population
    
    def fitness(chromo):
        def calc(v, c):
            return (c[0] * v**2) + (c[1] * v) + c[2]
        err = (calc(-1.0, target_chromosome) - calc(-1.0, chromo)) ** 2
        err += (calc(0, target_chromosome) - calc(0, chromo)) ** 2
        err += (calc(1.0, target_chromosome) - calc(1.0, chromo)) ** 2            
        err += (calc(3.0, target_chromosome) - calc(3.0, chromo)) ** 2 
        return err
    
    def end_below_tenth(statistics):
        if len(statistics) == 0:
            return False
        return statistics[-1].best_score < 0.005

    # chromo_a = ContinuousChromosome([0, 1, 2, 3, 4])
    # chromo_b = ContinuousChromosome([9, 8, 7, 6, 5])
    # for _ in range(15):
    #     print(crossover(chromo_a, chromo_b))
        
    
    population = generate_population()
    ga = ContinuousGeneticAlgorithm(population, fitness, ending=end_below_tenth)
    stats = ga.run(max_generations=1000, sd=0.2)

    print('Generations Run = {}'.format(len(stats)))
    print('Target Individual = {}'.format(target_chromosome)) 
    print('Best Score = {}'.format(stats[-1].best_score))       
    print('Best Individual = {}'.format(stats[-1].best_individual))
    
    plot_ga_scores(stats)
    
    plt.show()    