import numpy as np
import pandas as pd
from os.path import join
import glob
import tensorflow as tf
import tensorflow_datasets as tfds

class CICIDSDataset(object):

    @classmethod
    def get_dataset(cls):
        X, y = cls.load_data()
        normalized_x = cls.normalize(X)
        tfds.Split(normalized_x)
        return cls.balancing_data(cls.split(cls.load_data()))

    @classmethod
    def load_data(cls):
        filenames = [i for i in glob.glob(join("/data.datasets/CICIDS2017", "*.pcap_ISCX.csv"))] #경로 확장용
        combined_csv = pd.concat([pd.read_csv(f,dtype=object) for f in filenames],sort=False)

        data = combined_csv.rename(columns=lambda x: x.strip()) #strip() : delete \n & space
        df_label = data['Label']
        data = data.drop(columns=['Flow Packets/s','Flow Bytes/s','Label']) #github
        nan_count = data.isnull().sum().sum()

        if nan_count>0:
            data.fillna(data.mean(), inplace=True) #replace Nan
        data = data.astype(float).apply(pd.to_numeric)

        X = data.values
        y = cls.encode_label(df_label.values)
        return X, y

    @classmethod
    def balancing_data(cls, X, y, seed): # X_train,y_train = balance_data(X_train,y_train,seed=SEED)
        np.random.seed(seed)
        unique,counts = np.unique(y,return_counts=True)
        mean_samples_per_class = int(round(np.mean(counts)))
        N,D = X.shape
        new_X = np.empty((0,D))
        new_y = np.empty((0),dtype=int)
        for i,c in enumerate(unique):
            temp_x = X[y==c]
            indices = np.random.choice(temp_x.shape[0],mean_samples_per_class)
            new_X = np.concatenate((new_X,temp_x[indices]),axis=0)
            temp_y = np.ones(mean_samples_per_class,dtype=int)*c
            new_y = np.concatenate((new_y,temp_y),axis=0)

        indices = np.arange(new_y.shape[0])
        np.random.shuffle(indices)
        new_X =  new_X[indices,:]
        new_y = new_y[indices]
        return new_X, new_y

    @classmethod
    def encode_label(cls, Y_str):
        labels_d = cls.make_value2index(np.unique(Y_str))
        Y = [labels_d[y_str] for y_str  in Y_str]
        Y = np.array(Y)
        return np.array(Y)

    @classmethod
    def make_value2index(cls, attacks):
        #make dictionary
        attacks = sorted(attacks)
        d = {}
        counter=0
        for attack in attacks:
            d[attack] = counter
            counter+=1
        return d

    @classmethod
    def normalize(cls, data):
        data = data.astype(np.float32)

        eps = 1e-15

        mask = data==-1
        data[mask]=0
        mean_i = np.mean(data,axis=0)
        min_i = np.min(data,axis=0)
        max_i = np.max(data,axis=0)

        r = max_i-min_i+eps
        data = (data-mean_i)/r

        data[mask] = 0
        return data
