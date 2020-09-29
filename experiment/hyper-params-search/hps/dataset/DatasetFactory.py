# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.dataset.MNISTDataset import MNISTDataset
from hps.dataset.DGA import DGA

# class : DatasetFactory
class DatasetFactory(object):
    @staticmethod
    def create(data_nm, dim=1):
        data_nm = data_nm.lower()
        if data_nm == "mnist":
            if dim == 1:
                return MNISTDataset.get_tf_dataset_1d()
        elif data_nm == "dga":
            return DGA.get_dataset()

if __name__ == '__main__':
    name = "dga"
    ds_train, ds_test = DatasetFactory.create(name)
    print(ds_train, ds_test)