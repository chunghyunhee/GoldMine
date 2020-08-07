import random
import numpy as np
import time
import sys
import math
import copy
import abc # for abstract method

from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class SimulatedAnnealing(HPOptimizationAbstract):
    def __init__(self, **kwargs):
        # inheritance init
        super(SimulatedAnnealing, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False

    def _check_hpo_params(self):
        # default parm input
        self._n_updates = self._hpo_params["n_updates"]
        self._T_max = self._hpo_params["Tmax"]
        self._T_min = self._hpo_params["Tmin"]
        self._n_steps = self._hpo_params["n_steps"]
        self._copy_strategy = self._hpo_params["copy_strategy"]
        self._start = self._hpo_params["start"]
        self._acceptance_prob = self._hpo_params["acceptance_prob"]
        # placeholders
        self._best_state = self._hpo_params["best_state"]
        self._best_energy = self._hpo_params["best_energy"]
        self._state = self._hpo_params["state"]

    def copy_state(self, state):
        # use copy.deepcopy
        if self._copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        # use list slices
        elif self._copy_strategy == 'slice':
            return state[:]
        # use state's copy
        elif self._copy_strategy == 'method':
            return state.copy()
        else :
            raise RuntimeError('')

    ## create a state change
    @abc.abstractmethod
    def move(self):
        pass

    ## cal state energy
    @abc.abstractmethod
    def energy(self):
        pass

    ## update new state
    def update(self, *args, **kwargs):
        self.default_update(*args, **kwargs)

    ## update, print the state
    def default_update(self, step, T, E, acceptance, improvement):
        elapsed = time.time() - self._start
        if step == 0:
            print('\n Temperature | Energy | Accept | Improve | Elapsed | Remaining ',
                  file=sys.stderr)
            print('\r{Temp:12.5f}  {Energy:12.2f}'
                  .format(Temp=T,Energy = E),
                  file=sys.stderr, end="")
            sys.stderr.flush()
        else:
            remain = (self._n_steps - step) * (elapsed / step)
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%} '
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement),
                  file=sys.stderr, end="")
            sys.stderr.flush()

    ## min energy by annealing
    def _annealing(self):
        # default
        step = 0
        self._start = time.time()

        # precompute factor for exp cooling from Tmax to Tmin
        if self._T_min <= 0.0:
            raise Exception("Exponential cooling requires a min")
        Tfactor = - math.log(self._T_max / self._T_min)

        # initial state
        T = self._T_max
        E = self.energy()
        prevState = self.copy_state(self._state)
        prevEnergy = E
        self._best_state = self.copy_state(self._state)
        self._best_energy = E
        trials, accepts, improves = 0, 0, 0

        if self._n_updates > 0:
            updateWavelength = self._n_steps / self._n_updates
            self.update(step, T, E, None, None)

        # move to the new states
        while step < self._n_steps :
            step += 1
            T = self._T_max * math.exp(Tfactor * step / self._n_steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else :
                E += dE
            trials += 1

            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # save prev state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else :
                # accept new state and compare to best state
                accepts += 1
                if dE < 0.0 :
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self._best_energy :
                    self._best_state = self.copy_state(self.state)
                    self._best_energy = E

            if self._n_updates > 1:
                if (step // updateWavelength) > ((step-1) // updateWavelength):
                    self.update(step, T, E, accepts/trials, improves/trials)
                    trials, accepts, improves = 0, 0, 0
        self.state = self.copy_state(self._best_state)

        return self._best_state, self._best_energy



if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
                "n_updates" : 0.5,
                "T_max" : 0.5,
                "T_min" : 0.5,
                "n_steps" : 0.25,
                "copy_strategy" : 0.25,
                "n_steps" : 10,
                "start" : 10,
                "acceptance_prob" : 5,
                "eval_key" : "accuracy"
            },
        "ml_params":{
            "model_param":{
                "input_units" : "100",
                "output_units" : "1",
                "global_step" : "10",
                "early_type" : "2",
                "min_step" : "10",
                "early_key" : "accuracy",
                "early_value" : "0.98",
                "method_type" : "Basic",
                "global_sn" : "0",
                "alg_sn" : "0",
                "algorithm_type" : "classifier",
                "job_type" : "learn"
            },
            "pbounds":{
                "dropout_prob": [0, 0.5],
                "optimizer_fn": ["Adam", "rmsprop", "Adadelta"],
                "learning_rate": [0, 0.8],
                "act_fn": ["Tanh", "ReLU", "Sigmoid"],
                "hidden_units" : [3,1024]
            }
        }
    }
    sa = SimulatedAnnealing(hprs_info=hprs_info)
    best_params = sa._annealing()
    print(best_params)














