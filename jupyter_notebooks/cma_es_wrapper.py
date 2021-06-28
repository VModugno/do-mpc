import numpy as np
import cma
import re
import math

class CMA_ES_Wrapper:

    def __init__(self, L = 0.5, delta_t = 0.01, params_num = 1, scenarios_num = 3):
        self.params_num = params_num
        self.scenarios_num = scenarios_num
        self.L = L
        self.delta_t = delta_t

    def get_mean_sigma(self, param_set):
        min_param = min(param_set)
        max_param = max(param_set)

        mean = np.mean(param_set)
        sigma = np.std(param_set)

        return mean, sigma

    def generate_step_history(self, r, mpc_history):
        hist = []
        for elem in mpc_history:
            x0, y0, theta0 = elem['x']
            wl, wr = elem['u']

            v = (wr + wl) * r / 2      
            w = (wr - wl) * r / self.L 

            x = x0 + v * self.delta_t * math.cos(theta0)
            y = y0 + v * self.delta_t * math.sin(theta0)
            theta = theta0 + w * self.delta_t
            
            hist.append([x,y,theta])
        return hist

    def loss(self, param_val):
        step_hist = self.generate_step_history(param_val[0], self.mpc_history)
        return np.sqrt(np.mean((np.array(self.step_mpc_history) - np.array(step_hist)) ** 2))

    def optimize(self, param_set, mpc_history):
        self.mpc_history = mpc_history
        self.step_mpc_history = [elem['final_x'] for elem in mpc_history]

        mean, sigma = self.get_mean_sigma(param_set)

        es = cma.CMAEvolutionStrategy((self.params_num+1)*[mean], sigma, {'verbose': -3,'popsize': self.scenarios_num})
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.loss(param_val) for param_val in solutions])
        es.stop()

        return es.result.xbest[0]

    def optimize_step(self, param_set, mpc_history):
        self.mpc_history = mpc_history
        self.step_mpc_history = [elem['final_x'] for elem in mpc_history]

        mean, sigma = self.get_mean_sigma(param_set)

        es = cma.CMAEvolutionStrategy((self.params_num+1)*[mean], sigma, {'verbose': -3,'popsize': self.scenarios_num})
        while not es.stop() and es.best.f > 1e-5:
            solutions = es.ask()
            es.tell(solutions, [self.loss(param_val) for param_val in solutions])
        es.stop()
        new_param_set = sorted([float(sample[0]) for sample in es.ask()], key=float)
        return new_param_set


