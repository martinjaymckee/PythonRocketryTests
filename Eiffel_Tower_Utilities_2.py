import math
import numbers
import random


import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate


import pyrse.continuous_ga
import pyrse.engines
import pyrse.environment
import pyrse.numpy_utils
import pyrse.pad
import pyrse.rocket_components
import pyrse.simulator
import pyrse.simulator_analysis
import pyrse.triggers
import pyrse.utils


class EiffelTowerModel(pyrse.rocket_components.Component):
    # TODO: NEED TO ADD PIECEWISE LINEAR CD CALCULATION TO THIS MODEL AND FIRST TEST IT WITH
    #   CONSTANT CD (IT SHOULD GIVE THE SAME RESULTS), THEN SET IT UP SO THAT IT CAN BE CONFIGURED
    #   WITH MULTIPLE POINTS
    def __init__(self, cd, frontal_area, empty_mass, engine, Re_max=5e6):
        pyrse.rocket_components.Component.__init__(self, calc_aero=True)
        self.cd = cd
        self.__frontal_area = frontal_area
        self.__L_ref = .8 # m -- TODO: THIS NEEDS TO BE CONFIGURED BASED ON THE FRONTAL AREA....        
        self.__empty_mass = empty_mass
        self.__engine = engine
        self.add(engine)
        self.__Re_max = Re_max
        self.__maximum_Re_seen = None
        
    @property
    def cd(self):
        return self.__cd
    
    @cd.setter
    def cd(self, _cd):
        self.__cd = _cd
        if isinstance(_cd, numbers.Number):
            self.__cd_is_constant = True
        else:
            self.__cd_is_constant = False # TODO: PROCESS PIECEWISE LINEAR CD
        return self.__cd
    
    @property
    def Re_max(self):
        return self.__Re_max
    
    def calc_mass(self, t0):
        return self.__empty_mass

    def get_maximum_Re(self):
        return self.__maximum_Re_seen
    
    def calc_cd(self, v, env):
        v_mag = pyrse.numpy_utils.magnitude(v.ecef)        
        Re = v_mag * self.__L_ref / env.kinematic_viscosity # Calculate Reynolds Number
        if (self.__maximum_Re_seen is None) or (Re > self.__maximum_Re_seen):
            self.__maximum_Re_seen = Re        
        if self.__cd_is_constant:
            return self.__cd
        return self.__cd(Re)


    @property
    def frontal_area(self):
        return self.__frontal_area
        


class EiffelTowerChromosome(pyrse.continuous_ga.ContinuousChromosome):
    def __init__(self, ps=[], Re_max=5e6, cd_min=0.05, cd_max=5.0):
        super().__init__(ps, Re_max=Re_max, cd_min=cd_min, cd_max=cd_max)
        ps = [float(p) for p in ps]
        Re_max = float(Re_max)
        Re_step = float(Re_max) / (len(ps) - 1)
        self.__Re_min = Re_step / 2
        self.__Re_max = Re_max - (Re_step / 2)
        self.__xp = np.linspace(self.__Re_min, self.__Re_max, len(ps))
        self.__interp = scipy.interpolate.interp1d(self.__xp, ps)
        self.__cd_min = cd_min
        self.__cd_max = cd_max
        
    @property
    def xp(self):
        return self.__xp

    @property
    def Re_max(self):
        return self.__Re_max
    
    def __call__(self, Re):
        if Re < 0:
            raise Exception('Error in EiffelTowerChromosole.__call__: Negative Reynolds Number passed in!')
            
        if Re >= self.__Re_max: # Saturate beyond maximum Re
            return self[-1]
        elif Re <= self.__Re_min:
            return self[0]
        
        return self.__interp(Re)
    
    def mutate_with_p(self, p, sd=None):
        for idx in range(len(self)):
            if p < random.uniform(0, 1):
                if sd is None:
                    self.__yp[idx] = random.uniform(self.__cd_min, self.__cd_max)
                else:
                    offset = False
                    while not offset:
                        cd = random.gauss(self[idx], sd)
                        if self.__cd_min <= cd <= self.__cd_max:
                            offset = True
                            self[idx] = cd
        ps = list(self)
        self.__interp = scipy.interpolate.interp1d(self.__xp, np.array(ps))
        return True
    
    

class SimpleEiffelTowerChromosome(pyrse.continuous_ga.ContinuousChromosome):
    def __init__(self, ps=[], Re_max=5e6, cd_min=0.05, cd_max=5.0):
        super().__init__(ps, Re_max=Re_max, cd_min=cd_min, cd_max=cd_max)
        self.__Re_max = float(Re_max)
        ps = [float(p) for p in ps]
        self.__cd_min = cd_min
        self.__cd_max = cd_max

    def __call__(self, Re):
        if Re < 0:
            raise Exception('Error in EiffelTowerChromosole.__call__: Negative Reynolds Number passed in!')
            
        return max(self.__cd_min, min(self.__cd_max, self[0] * Re + self[1]))
    
    # def mutate_with_p(self, p, sd=None):
    #     for idx in range(len(self)):
    #         if p < random.uniform(0, 1):
    #             if sd is None:
    #                 self.__yp[idx] = random.uniform(self.__cd_min, self.__cd_max)
    #             else:
    #                 offset = False
    #                 while not offset:
    #                     cd = random.gauss(self[idx], sd)
    #                     if self.__cd_min <= cd <= self.__cd_max:
    #                         offset = True
    #                         self[idx] = cd
    #     ps = list(self)
    #     self.__interp = scipy.interpolate.interp1d(self.__xp, np.array(ps))
    #     return True
    
    

def create_linear_population(N, M=21, cd_min=0.05, cd_max=5, cd_sd=None, Re_max=5e6, simple=False):
    population = []
    cd_sd = (cd_max - cd_min) / 20 if cd_sd is None else cd_sd
    for _ in range(N):
        chromo = None
        cd0 = random.uniform(cd_min, cd_max)
        cd1 = random.uniform(cd_min, cd0) # Ensure a decreasing function
        m = (cd1 - cd0) / Re_max
        if simple:
            chromo = SimpleEiffelTowerChromosome([m, cd0], Re_max=Re_max, cd_min=cd_min, cd_max=cd_max)
        else:
            points = []
            for Re in np.linspace(0, Re_max, M):
                cd = m*Re + cd0
                offset = False
                cd_offset = None
                while not offset:
                    cd_offset = random.gauss(cd, cd_sd)
                    if cd_min <= cd_offset<= cd_max:
                        offset = True
                points.append(cd_offset)
            chromo = EiffelTowerChromosome(points, Re_max=Re_max, cd_min=cd_min, cd_max=cd_max)
        population.append(chromo)
    return population

    
def create_model(cd, engine, m_additional=0, scale=1.0):
    base_weight = 0.26 * pow(scale, 2.5)
    ref_area = pow(.237*scale, 2)
    return EiffelTowerModel(cd, ref_area, base_weight+m_additional, engine)
                            

def test_engine_randomizer(engine_randomizer, N=10, M=50):
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    for _ in range(N):
        eng = engine_randomizer()
        eng.start(0)
        t_burnout = eng.burn_time
        ts = np.linspace(0, t_burnout, M)
        Ts = np.array([eng.thrust(t) for t in ts])
        ax.plot(ts, Ts)
        

def create_test_set(flights, oversample=10, altitude_noise_sd=0.1):
    engs = []
    alts = []
    positions = []
    
    for pos, eng, alt in flights:
        engine_randomizer = pyrse.engines.EngineRandomizer(eng)
        for _ in range(oversample):
            positions.append(pos)
            engs.append(engine_randomizer())
            alts.append(random.gauss(alt, altitude_noise_sd))
    # print(positions)
    # print([eng.total_impulse for eng in engs])
    # print(alts)
    return positions, engs, alts


def create_engine_set(eng, oversample=10):
    engs = []
    engine_randomizer = pyrse.engines.EngineRandomizer(eng)
    for _ in range(oversample):
        engs.append(engine_randomizer())
    return engs


def get_sim_apogee(sim):
    log = sim.logs[0]
    pos = sim.pad.pos
    pos_ecef = np.array([r.pos.ecef for r in log])
    pos_enu = np.array([pyrse.coordinates.ECEFToENU(p, pos.ecef) for p in pos_ecef])
    return np.max(pos_enu)
    

def estimate_cd(flights, cd_range=[.01, 10], bins=50, iterations=5, oversample=3, alt_err_tol=0.005):
    bins = max(bins, 3)
#    pos = pyrse.utils.GeographicPosition.LLH(38.155458, -104.808906, 1663.5)    
    positions, engs, alts = create_test_set(flights, oversample = oversample)
    cd_min, cd_max = cd_range
    cd_best = None
    alt_err = None
    last_alt_err = None
    print('Estimate CD:')
    for iteration in range(iterations):
        print('\tIteration {} of {} '.format(iteration + 1, iterations), end='')
        cd_bin_width = (cd_max - cd_min) / bins
        cds = np.linspace(cd_min, cd_max, bins)
        
        error_map = []
        for cd in cds:
            err = 0
            for pos, eng, alt in zip(positions, engs, alts):
                print('.', end='')
                pad = pyrse.pad.LaunchPad(pos) 
                env = pyrse.environment.Environment(pad.pos)                
                model = create_model(cd, eng)
                triggers = [pyrse.triggers.ApogeeTrigger(model, pyrse.triggers.SimActions.EndSim), pyrse.triggers.SimulationTimedEndTrigger(20)]    
                sim = pyrse.simulator.Simulation1D(env, pad, model, triggers=triggers)    
                sim_status = pyrse.simulator.RunSimulation(sim)
                h_apogee = get_sim_apogee(sim) # TODO: REPLACE WITH A SIM EXTRACTOR....
                err += pow(h_apogee - alt, 2)
            error_map.append( (math.sqrt(err) / len(engs), cd) )
        print('')
        error_map.sort()

        cd_best = error_map[0][1]
        cd_min = max(0, cd_best - cd_bin_width)
        cd_max = cd_best + cd_bin_width        
        alt_err = error_map[0][0]
        if last_alt_err is not None:
            if (abs(last_alt_err - alt_err) / alt_err) < alt_err_tol:
                break
        last_alt_err = alt_err
    return cd_best, alt_err


def estimate_cd_ga(flights, cd_range=[.01, 5], pop_size=150, max_generations=250, oversample=5, alt_err_tol=0.015, elitism=0.15, mutation_sd=0.075, monotonicity_weight=1.075, debug_file=None, num_saved=None):
    print('Estimate CD with GA:')
    num_flights = len(flights)
    positions, engs, alts = create_test_set(flights, oversample = oversample)
    cd_min, cd_max = cd_range

    population = create_linear_population(pop_size, cd_min = cd_range[0], cd_max = cd_range[1])
    
    @pyrse.continuous_ga.monotonic_constraint(monotonicity_weight)
    def fitness(chromosome):
        err = 0
        for pos, eng, alt in zip(positions, engs, alts):
            pad = pyrse.pad.LaunchPad(pos) 
            env = pyrse.environment.Environment(pad.pos)                
            model = create_model(chromosome, eng)
            triggers = [pyrse.triggers.ApogeeTrigger(model, pyrse.triggers.SimActions.EndSim), pyrse.triggers.SimulationTimedEndTrigger(20)]    
            sim = pyrse.simulator.Simulation1D(env, pad, model, triggers=triggers)    
            sim_status = pyrse.simulator.RunSimulation(sim)
            h_apogee = get_sim_apogee(sim) # TODO: REPLACE WITH A SIM EXTRACTOR....
            err += pow((h_apogee - alt) / alt, 2)
        return math.sqrt(err / num_flights)

    def end_below_tol(stats):
        num_check = 1 if num_saved is None else int(num_saved)
        if len(stats) == 0:
            return False
        for idx in range(num_check):
            if stats[-(idx+1)].best_score > alt_err_tol:
                return False
        return True

    mutation_sd_bins = 5
    mutation_sd_bin_width = int(max_generations / mutation_sd_bins)
    mutation_sd_min = mutation_sd / 10
    mutation_sd_step = (mutation_sd - mutation_sd_min) / (mutation_sd_bins - 1)
    def sd_func(idx, best, mean):
        idx = int(idx / mutation_sd_bin_width)
        sd = mutation_sd - (idx * mutation_sd_step)
        print('sd = {}'.format(sd))
        return sd
    
    ga_callbacks = {
        'begin_generation': lambda num: print('Starting generation {}'.format(num)),
        'calc_individual': lambda chromosome: print('.', end=''), 
        'end_generation': lambda num, stat: print('\n\tbest fitness = {:0.3f}'.format(stat.best_score))
    }
    
    ga_config = pyrse.continuous_ga.GeneticAlgorithmConfiguration(elitism=elitism, max_generations=max_generations)
    ga = pyrse.continuous_ga.ContinuousGeneticAlgorithm(population, fitness, ending=end_below_tol, callbacks=ga_callbacks, debug_file=debug_file)
    stats = ga.run(max_generations=max_generations, sd=sd_func, num_saved=num_saved)

    pyrse.continuous_ga.plot_ga_scores(stats)
    return stats[-1].best_individual, stats[-1].best_score, stats[-1].saved


def optimize_additional_weight(cd, eng, m_additional_range = [0, 0.5], bins = 5, iterations = 15, oversample = 25, alt_best_tol=0.005):
    bins = max(bins, 3)
    pos = pyrse.utils.GeographicPosition.LLH(38.155458, -104.808906, 1663.5)    
    engs = create_engine_set(eng, oversample = oversample)
    pad = pyrse.pad.LaunchPad(pos)
    env = pyrse.environment.Environment(pad.pos)
    m_additional_min, m_additional_max = m_additional_range
    alt_best = None
    last_alt_best = None
    print('Optimize Additional Mass:')
    for iteration in range(iterations):
        print('\tIteration {} of {} '.format(iteration + 1, iterations), end='')
        m_bin_width = (m_additional_max - m_additional_min) / bins
        m_additionals = np.linspace(m_additional_min, m_additional_max, bins)
        
        alt_map = []
        for m_additional in m_additionals:
            alt_sum = 0
            for eng in engs:
                print('.', end='')
                model = create_model(cd, eng, m_additional = m_additional)
                triggers = [pyrse.triggers.ApogeeTrigger(model, pyrse.triggers.SimActions.EndSim), pyrse.triggers.SimulationTimedEndTrigger(20)]    
                sim = pyrse.simulator.Simulation1D(env, pad, model, triggers=triggers)    
                sim_status = pyrse.simulator.RunSimulation(sim)
                h_apogee = get_sim_apogee(sim)
                alt_sum += h_apogee
            alt_map.append( (alt_sum / len(engs), m_additional) )
        print('')
        alt_map.sort(reverse = True)

        m_additional_best = alt_map[0][1]
        if m_additional_best < m_additional_range[0]:
            break
        m_additional_min = m_additional_best - m_bin_width
        m_additional_max = m_additional_best + m_bin_width        
        alt_best = alt_map[0][0]
        if last_alt_best is not None:
            if (abs(last_alt_best - alt_best) / alt_best) < alt_best_tol:
                break
        last_alt_best = alt_best
    return m_additional_best, alt_best


def test_with_all_engines(cd_est, engs=None, m_additional = 0.0, pos = None, num_displayed=10, h_tgt=331, scale=1.0):
    pos = pyrse.utils.GeographicPosition.LLH(38.155458, -104.808906, 1663.5) if pos is None else pos
    if engs is None:
        engs = []
        eng_dir = pyrse.engines.EngineDirectory('./Engines')
        for eng_ref in eng_dir.directory:
            eng = eng_dir.load_first(*eng_ref) # TODO: IMPLEMENT A SELECT METHOD TO LOAD ALL ENGINES THAT MEET CERTAIN CONSTRAINTS
            engs.append(eng)
    pad = pyrse.pad.LaunchPad(pos)
    env = pyrse.environment.Environment(pad.pos)
    
    extract_values = ['t', 'v_mag:magnitude(vel)', 'h:alt(pos)']
    log_extractor = pyrse.simulator_analysis.SimulationLogExtractor(extract_values)
    run_data = []
    
    for eng in engs:
        model = create_model(cd_est, eng, m_additional, scale=scale)
        triggers = [pyrse.triggers.ApogeeTrigger(model, pyrse.triggers.SimActions.EndSim), pyrse.triggers.SimulationTimedEndTrigger(20)]    
        sim = pyrse.simulator.Simulation1D(env, pad, model, triggers=triggers)    
        sim_status = pyrse.simulator.RunSimulation(sim)
        print('{}'.format(eng))
#        h_apogee = get_sim_apogee(sim)
        results = log_extractor(sim)[0]
        h_apogee = np.max(results.h)
        v_max = np.max(results.v_mag)
        Re_max = int(model.get_maximum_Re())
        print('\talt = {:0.1f} m, t_opt = {:0.2f} s, v_max = {:0.2f} m/s, Re_max = {:d}'.format(h_apogee, sim.logs[0][-1].t - eng.burn_time, v_max, Re_max))

        print()
        run_data.append( (h_apogee, eng, results) )
    
    run_data.sort(key=lambda a: a[0], reverse=True)
    
    num_displayed = min(num_displayed, len(run_data))
    # print('number to display = {}'.format(num_displayed))
    display_data = run_data[:num_displayed]
    # print('run_data = {}'.format(run_data[:num_displayed]))
    
    fig, axs = plt.subplots(2, constrained_layout=True)
    for h_apogee, eng, result in display_data:
        label = eng.model
        axs[0].plot(result.t, result.h, label=label)
        axs[1].plot(result.t, result.v_mag, label=label)

    if h_tgt is not None:
        axs[0].axhline(h_tgt, c='k')
        
    for ax in axs:
        ax.legend()
    
    
def parse_cd_estimates(path, estimate_mean=True, Re_max_override=5e6, plot=True):
    try:
        fig, ax = None, None
        estimates = []
        with open(path, 'r') as file:
            print('{} open!'.format(path))

            for line in file.readlines():
                tokens = [t.strip() for t in line.split(';')]
                run = int(tokens[0])
                individual = int(tokens[1])
                fitness = float(tokens[2])
                Re_max = Re_max_override if len(tokens) == 4 else float(tokens[3])
                cds_string = tokens[-1][1:-1]
                cds = [float(v.strip()) for v in cds_string.split(',')]
                chromosome = EiffelTowerChromosome(cds, Re_max)
                estimates.append( (run, individual, fitness, Re_max, chromosome))

        if plot:            
            fig, ax = plt.subplots(1, constrained_layout=True, figsize = (8, 7))
            fig.suptitle('Estimated $C_d$ vs. $Re$')
            ax.set_xlabel('Reynolds Number ($Re$)')
            ax.set_ylabel('Coefficent of Drag')

            xs = np.linspace(0, estimates[0][-1].Re_max, 50)
            
            for _, _, fitness, _, chromosome in estimates:
                ys = [chromosome(x) for x in xs]
                # ax.plot(xs, ys, alpha=0.25, label='{:0.4f}'.format(fitness))
                ax.plot(xs, ys, alpha=0.5)#, label='{:0.4f}'.format(fitness))                
            #ax.legend() 
                
        if estimate_mean and (len(estimates) > 0):
            cd_sum = np.array(estimates[0][4])
#            print(cd_sum.shape)
            for _, _, _, _, cd in estimates[1:]:
                cd_additional = np.array(cd)
                #print(cd_additional.shape)
                cd_sum += cd_additional
                #print(cd_sum)
            estimates = [EiffelTowerChromosome(list(cd_sum / cd_sum.shape[0]), estimates[0][4].Re_max)]
            
            #print(estimates[0])
            if plot:
                est_mean = estimates[0]
                ys = [est_mean(x) for x in xs]
                ax.plot(xs, ys, '.-k', alpha=1, lw=2)
        if fig is not None:
            fig.savefig(r'D:\Workspace\Work\Apogee\Articles\Eiffel Tower Altitude\Images\part_I_ga_curve_fit.png', dpi=300)        
        return estimates
    except Exception as e:
        print(e)
    return None


def main(pos = pyrse.utils.GeographicPosition.LLH(38.155458, -104.808906, 1663.5)):
    N = 1 # Runs
    M = 20 # Best Individuals
    
    estimates = parse_cd_estimates('cd_est.dat.dat')
    for est in estimates:
        print(est)
    plt.show()
    return
    
    engs = pyrse.engines.EngineDirectory('./Engines')
    print(engs.directory)
    #engine = engs.load_first("Estes Industries, Inc.", "E16") 
    engine = engs.load_first("Estes A8-3", approx_match=True)     
    print(engine)
    quit()
#    engine_randomizer = pyrse.engines.EngineRandomizer(engine)
    print('engine mass = {} kg'.format(engine.mass(0)))
    print('engine mass fraction = {} %'.format(engine.mass_fraction))
    
#    test_engine_randomizer(engine_randomizer)
    score_pos = pyrse.utils.GeographicPosition.LLH(38.155458, -104.808906, 1663.5)
    slvroc_pos = pyrse.utils.GeographicPosition.LLH(37.57917, -106.148084, 2336) # NOTE: THIS IS ACTUALLY THE COORDINATES FOR MONTE VISTA
    
    flights = [
        (score_pos, engs.load_first("Estes Industries, Inc.", "E16"), 81/3.28),
        (score_pos, engs.load_first("At", "F67W"), 146/3.28),
        (slvroc_pos, engs.load_first("Aerotech", "G74W"), 348/3.28),
        (score_pos, engs.load_first("Aerotech", "G74W"), 285/3.28),
        (score_pos, engs.load_first("Aerotech", "G75J"), 338/3.28)
    ]
  
    predicted_engs = [
        engs.load_first("Aerotech", "H45"),
        engs.load_first("Aerotech", "I65W"),        
        engs.load_first("Aerotech", "J99")
    ]

    print('Test Engines = {}'.format([str(eng) for eng in predicted_engs]))

    fig, ax = plt.subplots(1, constrained_layout=True)
    xs = []
    ys = []
    ref_env = pyrse.environment.Environment(score_pos)
    ref_rho = ref_env.density
    for pos, eng, measured_alt in flights:
        flight_env = pyrse.environment.Environment(pos)
        corrected_alt = measured_alt * flight_env.density / ref_rho
        print('Total Impulse = {}, Measured Alt = {}, Corrected Alt = {}, rho = {}'.format(eng.total_impulse, measured_alt, corrected_alt, flight_env.density))
        xs.append(eng.total_impulse)
        ys.append(corrected_alt)
    ax.scatter(xs, ys, c='g')


    m, b = np.polyfit(xs, ys, 1)


    predicted_xs = []
    predicted_ys = []
    for eng in predicted_engs:
        impulse = eng.total_impulse
        predicted_alt = (m * impulse) + b
        predicted_xs.append(impulse)
        predicted_ys.append(predicted_alt)
        print('Predicted altitude with the {} {} is {} m ({} ft)'.format(eng.manufacturer, eng.model, predicted_alt, 3.28 * predicted_alt))
    ax.scatter(predicted_xs, predicted_ys, c='m')

    x_min, x_max = np.min(xs), np.max(xs)
    x_min = min(x_min, np.min(predicted_xs)), max(x_max, np.max(predicted_xs))
    print(x_min, x_max)
    plt.show()

    # population = create_linear_population(25)
    # fig, ax = plt.subplots(1, constrained_layout=True)
    
    # Res = np.linspace(0, 50000, 100)
    # for individual in population:
    #     ys = np.array([individual(Re) for Re in Res])
    #     ax.scatter(Res, ys)
        
 #    cd_est = 1.53
 #    alt_err = 2.4
 # #   cd_est, alt_err = estimate_cd(flights)
    # xs = np.linspace(0, 5e6, 50)
    # fig, ax = plt.subplots(1, constrained_layout=True)   
    # for run_num in range(N):
    #     with open('cd_est.dat', 'w+') as file:
    #         cd_est, alt_err, saved = estimate_cd_ga(flights, num_saved=M)
    #         print('cd = {}, alt_err = {}'.format(cd_est, alt_err))
    #         for idx, (fit, est) in enumerate(saved):
    #             file.write('{}; {}; {}; {}\n'.format(run_num, idx, fit, list(est)))
    #             alpha = max(0, min(1-fit, 1))
    #             ys = np.array([est(x) for x in xs])
    #             ax.plot(xs, ys, alpha=alpha, label='{:0.3f}'.format(fit))
    # ax.legend()
    
    #engine = engs.load_first("Estes Industries, Inc.", "E16") 
#    engine = engs.load_first("At", "F67W")
    #engine = engs.load_first("Aerotech", "H13ST")     
    #engine = engs.load_first("Aerotech", "I59WN-P")    
    #engine = engs.load_first("Aerotech", "J99")
    #engine = engs.load_first("Cti", "395-I55-MY-9A")
    #engine = engs.load_first("Aerotech", "I40N-P")
    # engine = engs.load_first('Cti', '1211-J140-WH_LB-P')
    #engine = engs.load_first('Cti', '644-J94-MY-P')
    #engine = engs.load_first('Aerotech', 'J90W')    
    # m_additional, alt_best = optimize_additional_weight(cd_est, engine)
    # print('m_additional = {}, alt = {}'.format(m_additional, alt_best))
    
#    engs = None    
#    engs = [engs.load_first("Estes Industries, Inc.", "E16"), engs.load_first("At", "F67W"), engs.load_first("Aerotech", "G74W")]
#    engs = [engs.load_first("Aerotech", "F67C"), engs.load_first("Aerotech", "G78"), engs.load_first("Aerotech", "G75M"), engs.load_first("Aerotech", "G74W"), engs.load_first("At", "F67W"), engs.load_first("Aerotech", "G80NBT"), engs.load_first("Aerotech", "G38FJ"), engs.load_first("Aerotech", "G40W"), engs.load_first("Aerotech", "F25"), engs.load_first("Aerotech", "F50T")]
#    engs = [engs.load_first("Estes Industries, Inc.", "E16"), engs.load_first('Cti', '168-H54-WH_LB-10A'), engs.load_first('Cti', 'G54-RL'), engs.load_first('Cesaroni', '108-G68-WH-13A'), engs.load_first("Aerotech", "H115DM-14A"), engs.load_first("Aerotech", "G77R"), engs.load_first("Aerotech", "F20"), engs.load_first("Aerotech", "F52C"), engs.load_first("Aerotech", "G74W"), engs.load_first("At", "F67W"), engs.load_first("Aerotech", "G78"), engs.load_first("Aerotech", "G75M"), engs.load_first("Aerotech", "G80NBT"), engs.load_first("Aerotech", "G38FJ"), engs.load_first("Aerotech", "G40W"), engs.load_first("Aerotech", "F50T")]
 

   # engs = [engs.load_first("Aerotech", "G74W")]

   #  cd_est = estimates[0]
   #  pos = score_pos
   #  for pos in [score_pos, slvroc_pos]:
   #      print('pos = {}'.format(pos))
   #      test_with_all_engines(cd_est, engs=engs, pos=pos, scale=1.0)
    
    # for results in log_results:
    #     print(dir(results))
    #     print(results.a)
    #     print(results.b)
    
#     model = create_model(0.708, engine)
#     pad = pyrse.pad.LaunchPad(pos)
#     env = pyrse.environment.Environment(pad.pos)
#     triggers = [pyrse.triggers.ApogeeTrigger(model, pyrse.triggers.SimActions.EndSim), pyrse.triggers.SimulationTimedEndTrigger(20)]
# # #    triggers = [pyrse.triggers.SimulationTimedEndTrigger(20)]


# #    model = EiffelTowerModel(.708, 13.6 / 100 / 100, 0.198, engine) # TEST MODEL IN OPENROCKET
#     print('initial position = {}'.format(pos))
    
#     sim = pyrse.simulator.Simulation1D(env, pad, model, triggers=triggers)    
#     sim_status = pyrse.simulator.RunSimulation(sim)

#     fig, axs = plt.subplots(3, constrained_layout = True, sharex = True)    
#     for log in sim.logs:
#         ts = np.array([r.t for r in log])
# #        print(ts)
#         Ts = np.array([pyrse.numpy_utils.magnitude(r.forces['T']) for r in log])
#         Ds = np.array([pyrse.numpy_utils.magnitude(r.forces['D']) for r in log])
#         ms = np.array([r.mass for r in log])
#         accels = np.array([r.accel.ecef for r in log])
#         Vs = np.array([r.vel.ecef for r in log])
#         pos_ecef = np.array([r.pos.ecef for r in log])
#         dpos_ecef_x = np.array([p[0] - pos.ecef[0] for p in pos_ecef])
#         dpos_ecef_y = np.array([p[1] - pos.ecef[1] for p in pos_ecef])
#         dpos_ecef_z = np.array([p[2] - pos.ecef[2] for p in pos_ecef])

#         pos_enu = np.array([pyrse.coordinates.ECEFToENU(p, pos.ecef) for p in pos_ecef])
#         pos_enu_e = np.array([p[0] for p in pos_enu])
#         pos_enu_n = np.array([p[1] for p in pos_enu])
#         pos_enu_u = np.array([p[2] for p in pos_enu])
        
#         gs = np.array([r.environment.g for r in log])
        
# #        Vs = np.array([r.vel.enu(pos) for r in log])
#   #       vxs = np.array([v.x for v in Vs])
#   #       vys = np.array([v.y for v in Vs])
#   #       vzs = np.array([v.z for v in Vs])


#         axs[0].plot(ts, Ds)
#         ax0 = axs[0].twinx()
#         ax0.plot(ts, Ts)
#         #axs[0].plot(ts, ms)
#         #axs[2].plot(ts, dpos_ecef_x, label='dp.x')
#         #axs[2].plot(ts, dpos_ecef_y, label='dp.y')
#         #axs[2].plot(ts, dpos_ecef_z, label='dp.z')
#         axs[1].plot(ts, Vs)
#         axs[2].plot(ts, pos_enu)
#         print('maximum altitude = {} m'.format(np.max(pos_enu_u)))
        
#         for ax in axs:
#             ax.legend()

#     print('final position = {}'.format(pos))
    plt.show()
    
    
if __name__ == '__main__':
    print('Run')
    main()