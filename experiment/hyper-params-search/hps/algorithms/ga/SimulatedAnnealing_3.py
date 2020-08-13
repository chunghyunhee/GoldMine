import numpy as np
from random import random
from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class SimulatedAnnealing(HPOptimizationAbstract):
    def __init__(self, **kwargs):
        # inheritance init
        super(SimulatedAnnealing, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False


    def _check_hpo_params(self):
        result_param_list = list()
        self._n_pop = 1
        self._k = 0.1
        self._M = self._hpo_params["M"]
        self._T0 = self._hpo_params["T0"]
        self._alpha = self._hpo_params["alpha"]

    # neighbor selection
    def _generate(self, param_list, score_list):
        # 기준점 : x0
        result_param_list = list()
        x0 = hprs_info["pbounds"]["dropout_prob"]

        for i in range(self._M):
            xt = 0
            result_param_list = list()

            # to make candidate : xt
            ran_x_1 = np.random.rand()
            ran_x_2 = np.random.rand()

            if ran_x_1 >= 0.5:
                x1 = self._k*ran_x_2
            else:
                x1 = -self._k*ran_x_2

            xt = x0 + x1

            num_result_params = len(result_param_list)
            result_param_list += self._generate_param_dict_list(num_result_params)
            return result_param_list

    def accept(self, param_dict_list, result_param_list):
        temp = []
        best_params_list = list()

        # dnn acc, score
        of_new = self._update(self.hash_idx_list, self.score_list)
        of_final = self._update(self.hash_idx_list, self.score_list)

        # 같으면 form을 확인하여 선택지 결정, 다르면 이웃을 선택한다.
        if param_dict_list == result_param_list :
            best_params_list += param_dict_list
        else :
            ran_1 = np.random.rand()
            form = 1 / (np.exp((of_new[1] - of_final[1]) / self._T0))
            if ran_1 <= form:
                best_params_list += result_param_list
            else :
                best_params_list += param_dict_list

        temp = np.append(temp, self._T0)
        self._T0 = self._alpha * self._T0

        return best_params_list

if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
                "beta" : 1.3,
                "T0" : 0.40,
                "alpha" : 0.85,
                "update_n" : 5,
                "n_steps": 10,
                "n_params": 10,
                "k_val": 1,
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
                #"optimizer_fn": ["Adam", "rmsprop", "Adadelta"],
                #"learning_rate": [0, 0.8],
                #"act_fn": ["Tanh", "ReLU", "Sigmoid"],
                #"hidden_units" : [3,1024]
            }
        }
    }
    ga = SimulatedAnnealing(hps_info = hprs_info)
    best_params = ga._generate([], [])
    print(best_params)