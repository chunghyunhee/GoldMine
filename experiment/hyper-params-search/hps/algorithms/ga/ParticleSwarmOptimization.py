import numpy as np
from random import random
from random import uniform
from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract


class ParticleSwarmOptimization(HPOptimizationAbstract):
    def __init__(self, **kwargs):
        # inheritance init
        super(ParticleSwarmOptimization, self).__init__(**kwargs)
        self._check_hpo_params()

    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._n_steps = self._hpo_params["n_steps"]
        self._c1 = self._hpo_params["c1"]     # cognitive constants
        self._c2 = self._hpo_params["c2"]     # social constants

    # generate candidate function
    def _generate(self, param_list, score_list):

        result_param_list = list()
        p_best_list = list()

        # generate random hyperparameter
        best_param_list = self._particle(param_list)
        p_best = self._p_best(best_param_list, score_list)
        p_best_list.append(p_best)
        g_best = self._g_best(best_param_list, p_best_list)
        #self.LOGGER.info("{},{}".format(p_best_list, g_best))
        self.check_param(p_best_list,g_best)
        compute_velocity_params = self.compute_velocity(best_param_list, p_best)
        update_position_params = self.update_position(best_param_list, compute_velocity_params)

        #check params
        result_param_list += update_position_params
        return result_param_list



    # to check updates
    def check_param(self, p_best_list, g_best):
        self.LOGGER.info("{}, {}, {}".format(p_best_list, g_best))

    def _p_best(self, param_list, score_list):
        if len(score_list) == 0:
            return param_list[0]
        else :
            max_score_value = max(score_list)

            for i in range(len(score_list)):
                if max_score_value == score_list[i]:
                    return param_list[i]


    def _g_best(self, param_list, p_best_list):
        if len(p_best_list) == 0:
            return param_list[0]
        else:
            global_value = max(p_best_list)
            for i in range(len(p_best_list)):
                return param_list[i]


    # random init particle position
    def _particle(self, param_list):
        if len(param_list) == 0:
            return self._generate_param_dict_list(self._n_pop)
        else :
            return param_list

    def compute_velocity(self,param_dict_list, pos_best_i):
        w = uniform(0,1)
        velocity_i = list()

        # randomly initialize velcocity list
        for j in range(len(param_dict_list)):
            velocity_i.append(uniform(-1, 1))

        for i, param_dict in enumerate(param_dict_list):
            for j in param_dict.keys():
                r1 = random()
                r2 = random()

                if type(param_dict[j]) == int or type(param_dict[j]) == float:
                    vel_cognitive = self._c1*r1*(pos_best_i[j] - param_dict[j])
                    vel_social = self._c2*r2*(pos_best_i[j] - param_dict[j])
                    velocity_i[i] = w * velocity_i[i] + vel_cognitive + vel_social

                else :
                    vel_cognitive = self._c1 * r1
                    vel_social = self._c2 * r2
                    velocity_i[i] = w * velocity_i[i] + vel_cognitive + vel_social

        return velocity_i


    # update position based on updated velocity
    def update_position(self, param_list, velocity_i):

        for i, param_dict in enumerate(param_list):
            for j in param_dict.keys():
                if type(param_dict[j]) == int or type(param_dict[j]) == float:
                    param_dict[j] = param_dict[j] + velocity_i[i]
                    # 범위 설정
                    min = self._pbounds[j][0]
                    max = self._pbounds[j][1]
                    param_dict[j] = np.clip(param_dict[j], min, max)
                # categorical 변수의 경우
                else :
                    param_dict[j] = param_dict[j]
        return param_list


# main __init__ to execute in this single file
if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
            "n_params" : 5,
            "n_steps" : 5,
            "c1": 3,
            "c2": 3,
            "k_val": 5,
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