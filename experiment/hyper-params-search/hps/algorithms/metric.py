import time
import numpy as np
import matplotlib.pyplot as plt
from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class metric(HPOptimizationAbstract):
    def __init__(self, params, **kwargs):
        # input variables
        self.params = params
        self.labels = self.params["labels"]
        self.normalization = self.params["normalization"]
        self.sample_weight = self.params["sample_weights"]  # acc score 기준 multiclass parameter

    ## multiclass 기준
    def _confusion_matrix(self, y_true, y_pred):
        # label
        self.labels = np.asarray(self.labels)
        n_labels = self.labels.size

        if n_labels == 0 :
            raise ValueError()
        elif y_true.size == 0 :
            return np.zeros((n_labels, n_labels), dtype = np.int)
        elif np.all([1 not in y_true for 1 in self.labels]):
            raise ValueError()

        # weight, normalization
        if self.normalization is None :
            return 1- self.score_list
        else :
            if self.sample_weight is None :
                self.sample_weight = np.ones(y_true.shape[0], dtype = np.int64)
            else :
                self.sample_weight = np.asarray(self.sample_weight)

        # TP, TN, FP, FN 계산
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if self.sample_weight is not None :
            tp_bins_weights = np.asarray(self.sample_weight)[tp]
        else :
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(tp_bins, weights= tp_bins_weights, minlength=len(self.labels))
        else :
            true_sum = pred_sum = tp_sum = np.zeros(len(self.labels))
            if len(y_pred):
                pred_sum = np.bincount(y_pred, weights = self.sample_weight, minlength=len(self.labels))
            if len(y_true):
                true_sum = np.bincount(y_true, weights=self.sample_weight, minlength = len(self.labels))

            indices = np.searchsorted(self.labels, self.labels[:n_labels])
            tp_sum = tp_sum[indices]
            true_sum = true_sum[indices]
            pred_sum = pred_sum[indices]

        fp = pred_sum - tp_sum
        fn = true_sum - tp_sum
        tp = tp_sum

        sample_weight =  np.array(self.sample_weight)
        self.tp = np.array(tp)
        self.fp = np.array(fp)
        self.fn = np.array(fn)
        self.tn = sample_weight * y_true.shape[1] - tp - fp - fn

        recall = tp / (tp+fn)
        return np.array([self.tn, self.fp, self.fn, self.tp]).T.reshape(-1, 1, 2)

    ## get precision and recall for each class
    def _recall(self):
        return self.tp / (self.tp + self.fn)

    def _precision(self):
        return self.tp / (self.tp + self.fp)

    def _f1_score(self):
        return 2* (self._precision() * self._recall()) / ( self._precision() + self._recall() )

    def roc_auc_score(self):
        # for N classes, N roc curve ( one vs all )
        for i in range(len(self.labels)):
            fpr = self.fp / (self.tn + self.fp)
            tpr = self.tp / (self.tp + self.fn)
            # plotting
            plt.plot(fpr, tpr)





















