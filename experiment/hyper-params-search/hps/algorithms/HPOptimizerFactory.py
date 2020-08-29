# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.

from hps.algorithms.ga.GeneticAlgorithm import GeneticAlgorithm
from hps.algorithms.ga.ParticleSwarmOptimization import ParticleSwarmOptimization

# class : HPOptimizerFactory
class HPOptimizerFactory(object):
    @staticmethod
    def create(hpo_dict):
        hpo_alg = hpo_dict["hpo_alg"]
        if hpo_alg == "GA":
            # TODO : Check init & remove get_ga_params
            ga = GeneticAlgorithm(hps_info=hpo_dict)
            return ga
        elif hpo_alg == "PSO":
            pso = ParticleSwarmOptimization(hps_info=hpo_dict)
            return pso
        else:
            raise NotImplementedError