import numpy as np
import time
import random

from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class SimulatedAnnealing(HPOptimizationAbstract):
    def __init__(self, **kwargs):
        # inheritance init
        super(SimulatedAnnealing, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False

    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._T0 = self._hpo_params["T0"]
        self._alpha = self._hpo_params["alpha"]
        self._n_steps = self._hpo_params["n_steps"]

    def _generate(self, param_list, score_list):
        result_param_list = list()

        best_param_list = self._init_param(param_list)            # 이전까지의 best값
        neighbor_list = self._neighbor_selection(best_param_list) # candidate
        result_param_list += best_param_list + neighbor_list

        if len(param_list) != 0 :
            result_param_list = self.accept(result_param_list)

        # duplicate check (최종 result_param_list내 dict값은 2개, n_pop값은 1)
        result_param_list = self._remove_duplicate_params(result_param_list)
        num_result_params = len(result_param_list)

        if num_result_params < 2 :
            result_param_list += self._generate_param_dict_list(1)
        elif num_result_params > 2 :
            random.shuffle(result_param_list)
            result_param_list = result_param_list[:2]

        return result_param_list

    def _init_param(self, param_list):
        if len(param_list) == 0:
            return self._generate_param_dict_list(self._n_pop)
        else :
            return param_list

    def _neighbor_selection(self, param_dict_list):
        neighbor_param_list = list()
        neighbor_param_dict = dict()

        for i, param_dict in enumerate(param_dict_list):
            if i == 0 :
                for j in param_dict.keys():
                    rand = np.random.random()

                    if type(param_dict[j]) == int :
                        x0 = param_dict[j]
                        min = self._pbounds[j][0]
                        max = self._pbounds[j][1]

                        if rand >= 0.5:
                            x1 = int(np.random.uniform(0, 30))
                        else:
                            x1 = int(np.random.uniform(-30, 0))
                        print(type(x0))
                        xt = np.clip(x0+x1, min, max)

                    elif type(param_dict[j]) == float :
                        x0 = param_dict[j]
                        min = self._pbounds[j][0]
                        max = self._pbounds[j][1]

                        if rand >= 0.5 :
                            x1 = np.random.uniform(0.0, 0.5)
                        else:
                            x1 = np.random.uniform(-0.5, 0.0)

                        xt = np.clip(x0 + x1, min, max)

                    else :
                        # best_param_list에 있는 index값 확인
                        if param_dict[j] == self._pbounds[j][0]:
                            index = 0
                        elif param_dict[j] == self._pbounds[j][1]:
                            index = 1
                        else:
                            index = 2

                        x0 = index
                        if rand >= 0.5:
                            x1 = int(np.random.uniform(0, 1))
                        else :
                            x1 = int(np.random.uniform(-1, 0))

                        xt = np.clip(x0+x1, 0, 2)  # index for string value in list
                    neighbor_param_dict[j] = xt
                neighbor_param_list.append(neighbor_param_dict)

        return neighbor_param_list



    # return list = [ best_param, candidate ]
    def accept(self, result_param_list):
        of_new = self.score_list[0]
        of_final = self.score_list[1]

        if of_new <= of_final :
            result_param_list[0] = result_param_list[1]
        else :
            ran_1 = np.random.rand()
            form = 1 / (np.exp((of_new - of_final) / self._T0))
            if ran_1 <= form:
                result_param_list[0] = result_param_list[1]

        # temperature
        self._T0 = self._alpha * self._T0

        return result_param_list


if __name__ == '__main__':
    hprs_info = {
        "hpo_params" :{
                "T0" : 0.40,
                "alpha" : 0.85,
                "n_steps" : 200,
                "n_pop" : 1,
                "k" : 0.1,
                "n_params": 1,
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