# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import sys, time
import json

from hps.common.Constants import Constants
from hps.common.Common import Common
from hps.algorithms.HPOptimizerFactory import HPOptimizerFactory

from hps.dataset.DatasetFactory import DatasetFactory
from hps.ml.TensorFlowAbstract import TensorFlowAbstract

# class : HPSearcher
class HPSearcher(object):
    def __init__(self, param_json_nm):
        self.LOGGER = Common.LOGGER.getLogger()
        f = open(Constants.DIR_PARAMS + "/" + param_json_nm, "r")
        param_str = f.read()
        f.close()
        self.hps_param_dict = json.loads(param_str)
        self.LOGGER.info(self.hps_param_dict)
        self.LOGGER.info("Hyper-Parameter Search Start...")
        self.best_param_dict_list = list()
        self.best_param_dict_list = list()

        self.best_hps_param_dict_list = list()


    def run(self):
        ## HPO Algorithm
        hpo_algorithm = HPOptimizerFactory.create(self.hps_param_dict)
        ## Optimize
        _, self.best_param_dict_list = hpo_algorithm.optimize()
        ## check
        #self.LOGGER.info("{}".format(self.best_param_dict_list))
        self.best_param_dict_list = hpo_algorithm.optimize()


    '''
    def predict(self):
        hpo_algorithm = HPOptimizerFactory.create(self.hps_param_dict)
        hpo_algorithm.optimize_test(self.best_param_dict_list)
        self.best_hps_param_dict_list = hpo_algorithm.optimize()[0] # 각 particle의 parameter 저장
    '''


    def predict(self, data_nm):
        ds_train, ds_test = DatasetFactory.create(data_nm)
        self.LOGGER.info("{}".format(self.best_hps_param_dict_list))
        predict_test = TensorFlowAbstract(self.best_hps_param_dict_list)
        predict_test.predict(ds_test)





if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage : python3.5 -m hps.main [param.json]")
    else :
        param_json_filename = sys.argv[1]
        hp_searcher = HPSearcher(param_json_filename)

        # train
        hp_searcher.run()
        # test

        #hp_searcher.predict()
        hp_searcher.predict()

        name = "MNIST"
        hp_searcher.predict(name)
        time.sleep(1)



