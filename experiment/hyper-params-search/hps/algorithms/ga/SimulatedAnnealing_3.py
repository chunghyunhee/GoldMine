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
        self._n_pop = self._hpo_params["n_pop"]
        self._M = self._hpo_params["M"]
        self._T0 = self._hpo_params["T0"]
        self._alpha = self._hpo_params["alpha"]

    # neighbor selection
    def _generate(self, param_list, score_list):
        # 기준점 : x0
        result_param_list = list()
        x0 = self._generate_param_dict_list(self._n_params)

        for i in range(self._M):
            xt = self._generate_param_dict_list(self._n_params)

            # to make candidate xt, make random value
            ran_x_1 = np.random.rand()

            # type에 따른 random값
            for _, (key, value) in enumerate(self._pbounds):
                if key == 'optimizer_fn' or key == 'act_fn':
                    min = 0
                    max = len(key)
                    xt = np.clip(x0, min, max)
                elif key == 'hidden_units' or key == "filter_sizes" or key == "pool_sizes":
                    min = value[0]
                    max = value[1]
                    xt = np.clip(x0, min, max)
                else :
                    min = value[0]
                    max = value[1]
                    if ran_x_1 >=0.5:
                        x1 = np.random.uniform(-0.1, 0.1)
                    else:
                        x1 = -np.random.uniform(-0.1, 0.1)
                    xt = np.clip(x0 + x1, min, max)

            result_param_list += xt
            return result_param_list

    def accept(self, param_dict_list, result_param_list, best_score_list, new_score_list):
        temp = []
        best_params_list = list()

        # dnn acc, score
        of_new = new_score_list
        of_final = best_score_list

        # best값과 neighbor값의 비교
        if of_new <= of_final :
            best_params_list = param_dict_list
        else :
            ran_1 = np.random.rand()
            form = 1 / (np.exp((of_new[1] - of_final[1]) / self._T0))
            if ran_1 <= form:
                best_params_list = result_param_list
            else :
                best_params_list = param_dict_list

        self._T0 = self._alpha * self._T0

        return best_params_list

if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
                "T0" : 0.40,
                "alpha" : 0.85,
                "n_pop" : 1,
                "k" : 0.1,
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
                "optimizer_fn": "Adam",
                "learning_rate": 0.8,
                "act_fn": "Sigmoid",
                "hidden_units" : 50
            }
        }
    }
    ga = SimulatedAnnealing(hps_info = hprs_info)
    best_params = ga._generate([], [])
    print(best_params)