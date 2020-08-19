import numpy as np
from random import random
from random import uniform
from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class ParticleSwarmOptimization:
    def __init__(self, **kwargs):
        # inheritance init
        super(ParticleSwarmOptimization, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False

    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._n_steps = self._hpo_params["n_steps"]
        self._position_list = self._hpo_params["position_list"]
        self._bounds = self._hpo_params(["bounds"])
        self._w = self._hpo_params(["w"])       # constant of the inertia weight
        self._c1 = self._hpo_params(["c1"])     # cognitive constants
        self._c2 = self._hpo_params(["c2"])     # social constants

    ## PSO overall process
    # generate candidate function
    def _generate(self, param_list, score_list):
        result_param_list = list()
        # generate random hyperparameter
        best_param_list = self._particle(param_list)
        update_velocity_params = self.velocity(param_list)
        update_position_params = self.position(param_list)

        result_param_list += update_velocity_params + update_position_params
        result_param_list = self._remove_duplicate_params(result_param_list)
        num_result_params = len(result_param_list)

        ## leak
        if  num_result_params < self._n_pop:
            result_param_list += self._generate_param_dict_list(self._n_pop - num_result_params)
        ## over
        elif num_result_params > self._n_pop :
            random.shuffle(result_param_list)
            result_param_list = result_param_list[:self._n_pop]

        return best_param_list

    def _particle(self, param_list):
        if len(param_list) == 0:
            return self._generate_param_dict_list(self._n_pop)
        else :
            return param_list

    # update new particle velocity
    def update_velocity(self, param_list):
        for i in range(0, len(param_list)):
            r1 = random()
            r2 = random()

            vel_cognitive = c1*r1*(self.pos_best_i[i] - self.position_i[i])
            vel_social = c2*r2*(param_list[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, param_list):
        for i in range(0, num_dimentions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust max position if necessary
            if self.position_i[i] > self.bounds[i][1]:
                self.position_i[i] = self.bounds[i][1]
            # adjust min position if necessary
            if self.position_i[i] < self.bounds[i][0]:
                self.position_i[i] = self.bounds[i][0]





