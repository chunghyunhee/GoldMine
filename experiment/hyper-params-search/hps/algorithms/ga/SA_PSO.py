## SA + PSO
import numpy as np
import random
import time

from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract
from hps.algorithms.ga.SimulatedAnnealing_3 import  SimulatedAnnealing

class SA_ParticleSwarmOptimization(SimulatedAnnealing, HPOptimizationAbstract):
    def __init__(self, **kwargs):
        # inheritance init
        super(SA_ParticleSwarmOptimization, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False

    def _check_hpo_params(self):
        self._n_pop = self._n_params

        ## PSO
        self._k = self._hpo_params["k"] # decide local optima
        self._w = self._hpo_params["w"]
        self._n_steps = self._hpo_params["n_steps"]
        self._c1 = self._hpo_params["c1"]     # cognitive constants
        self._c2 = self._hpo_params["c2"]     # social constants
        self._delta = self._hpo_params["delta"]    # modified PSO
        self.count = self._hpo_params["count"]

        ## SA
        self._T0 = self._hpo_params["T0"]
        self._alpha = self._hpo_params["alpha"]

    # generate candidate function
    def _generate(self, param_list, score_list, iter_num):
        result_param_list = list()
        p_best_list = list()

        # generate random hyperparameter
        best_param_list = self._particle(param_list)

        # pbest
        p_best = self._p_best(best_param_list, score_list)
        p_best_list.append(p_best)

        # gbest갱신
        g_best, p_best_list = self._g_best(best_param_list, p_best_list)
        # k번동안 update되지 않으면 sa로 새로 갱신
        g_best_pso = self.update_gbest(g_best)


        # position 변경
        compute_velocity_params = self.compute_velocity(best_param_list, p_best, g_best_pso)
        update_position_params = self.update_position(best_param_list, compute_velocity_params)
        result_param_list += update_position_params

        # if duplicate, generate new particle
        result_param_list = self._remove_duplicate_params(result_param_list)

        num_result_params = len(result_param_list)
        ## leak
        if num_result_params < self._n_pop:
            result_param_list += self._generate_param_dict_list(self._n_pop - num_result_params)
        ## over
        elif num_result_params > self._n_pop:
            random.shuffle(result_param_list)
            result_param_list = result_param_list[:self._n_pop]

        return result_param_list



    # 해당 iteration 중 모든 particle에서 최대
    def _p_best(self, param_list, score_list):
        if len(score_list) == 0:
            return param_list[0]
        else :
            max_score_value = max(score_list)
            for i in range(len(score_list)):
                if max_score_value == score_list[i]:
                    return param_list[i]

    # global에서 최대
    def _g_best(self, param_list, p_best_list):
        all_list = list()

        if len(p_best_list) == 0:
            all_list.append(param_list[0])
            return param_list[0], all_list
        else:
            global_value = max(p_best_list)
            for i in range(len(p_best_list)):
                if global_value == p_best_list[i]:
                    all_list.append(global_value)
                    return global_value, all_list

    # global value를 받아 sa진행
    def update_gbest(self, global_dict):
        self.count += 1 # 현제 step count

        if self.count % self._k == 0 :
            result_param_list = list()
            best_param_list = list()

            best_param_list.append(global_dict)
            neighbor_param_list = self._neighbor_selection(best_param_list)
            result_param_list += best_param_list + neighbor_param_list # ( glbal_best , neighbor_candidate )

            if len(self.score_list) != 0:
                result_param_list = self.accept(result_param_list)

            return result_param_list[0]

        else :
            result_param_list = list()
            result_param_list.append(global_dict)
            return result_param_list[0]


    # random init particle position
    def _particle(self, param_list):
        if len(param_list) == 0:
            return self._generate_stratified_param_list(self._n_pop)
        else :
            return param_list



    def compute_velocity(self,param_dict_list, pos_best_i, g_best_i):
        # initialize each velocity dictionary in list
        velocity_list = list()
        velocity_dict = dict()

        for _, key in enumerate(self._pbounds):
            velocity_dict[key] = random.uniform(-1, 1)
        for _ in range(self._n_pop):
            velocity_list.append(velocity_dict)

        for i, param_dict in enumerate(param_dict_list):
            for j in param_dict.keys():
                ## gbest값에 따라 parameter다르게 선정
                r1 = random.random()
                r2 = random.random()

                # modified velocity for multi-dim
                if type(param_dict[j]) == int or type(param_dict[j]) == float:

                    if (abs(velocity_list[i][j]) + abs(g_best_i[j] - param_dict[j]) < self._delta) and type(param_dict[j] == float):
                        velocity_list[i][j] =  (2*random.random()-1) * self._delta
                    else:
                        vel_cognitive = self._c1*r1*(pos_best_i[j] - param_dict[j])
                        vel_social = self._c2*r2*(g_best_i[j] - param_dict[j])
                        velocity_list[i][j] = self._w * velocity_list[i][j] + vel_cognitive + vel_social

                else :
                    vel_cognitive = self._c1 * r1
                    vel_social = self._c2 * r2
                    velocity_list[i][j] = self._w * velocity_list[i][j] + vel_cognitive + vel_social

        return velocity_list


    # update position based on updated velocity
    def update_position(self, param_list, velocity_i):

        for i, param_dict in enumerate(param_list):
            for j in param_dict.keys():
                if type(param_dict[j]) == int or type(param_dict[j]) == float:
                    param_dict[j] = param_dict[j] + velocity_i[i][j]
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
            "w" : 0.1,
            "delta" : 1,
            "n_params" : 10,
            "n_steps" : 20,
            "c1": 0.3,
            "c2": 0.3,
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