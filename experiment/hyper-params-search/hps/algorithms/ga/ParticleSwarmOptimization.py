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
        compute_velocity_params = self.compute_velocity(param_list)
        update_position_params = self.update_position(param_list)

        result_param_list += compute_velocity_params + update_position_params
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

    # compute velocity to update position of particle
    def compute_velocity(self, param_list, pos_best_i, position_i):
        # compute velocity of each particle's hyperparameter parameters
        velocity_i = list()
        for i in range(0, len(param_list)):
            r1 = random()
            r2 = random()

            vel_cognitive = self._c1*r1*(pos_best_i[i] - position_i[i])
            vel_social = self._c2*r2*(param_list[i] - position_i[i])

            velocity_i[i] = self._w * velocity_i[i] + vel_cognitive + vel_social

        return velocity_i

    # update the particle position based off new velocity updates
    def update_position(self, param_list, position_i, velocity_i):
        for i in range(0, len(param_list)):
            position_i[i] = position_i[i] + velocity_i[i]

            # adjust max position if necessary
            if position_i[i] > self._bounds[i][1]:
                position_i[i] = self._bounds[i][1]

            # adjust min position if necessary
            if position_i[i] < self._bounds[i][0]:
                position_i[i] = self._bounds[i][0]

        return position_i


# main __init__ to execute in this single file
if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
                "n_steps" : 100,
                "position_list" : [1,2,3,4],
                "bounds" : [0,10],
                "w" : 0.1,
                "c1": 10,
                "c2": 1,
                "eval_key": "accuracy"
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
                "optimizer_fn": "Adam",
                "learning_rate": 0.8,
                "act_fn": "Sigmoid",
                "hidden_units" : 50
            }
        }
    }
    pso = ParticleSwarmOptimization(hps_info = hprs_info)
    best_params = pso._generate([], [])

    print(best_params)



