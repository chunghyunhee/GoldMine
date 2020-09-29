# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.

from hps.algorithms.ga.GeneticAlgorithm import GeneticAlgorithm
from hps.algorithms.ga.ParticleSwarmOptimization import ParticleSwarmOptimization
from hps.algorithms.ga.SimulatedAnnealing_3 import SimulatedAnnealing
from hps.algorithms.ga.base_PSO import BASE_ParticleSwarmOptimization
from hps.algorithms.ga.SA_PSO import SA_ParticleSwarmOptimization
from hps.algorithms.ga.PSO_GA import GA_ParticleSwarmOptimization
from hps.algorithms.ga.PSO_boudary import ParticleSwarmOptimization_boundary

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
        elif hpo_alg == "SA":
            sa = SimulatedAnnealing(hps_info=hpo_dict)
            return sa
        elif hpo_alg == "base_PSO":
            base_pso = BASE_ParticleSwarmOptimization(hps_info = hpo_dict)
            return base_pso
        elif hpo_alg == "SA_PSO":
            sa_pso = SA_ParticleSwarmOptimization(hps_info= hpo_dict)
            return sa_pso
        elif hpo_alg == "GA_PSO":
            ga_pso = GA_ParticleSwarmOptimization(hps_info= hpo_dict)
            return ga_pso
        elif hpo_alg == "PSO_boundary":
            pso_boundary = ParticleSwarmOptimization_boundary(hps_info= hpo_dict)
            return pso_boundary
        else:
            raise NotImplementedError