import os
import numpy as np

class Loss(object):
    def __init__(self):
        pass

    def __call__(self):
        pass

class MSE(Loss):
    def __init__(self):
        pass

    def __to_categorical__(self, y_true):
        category_ = list()
        n_classes = len(np.unique(y_true))

        for y in y_true:
            category = np.eye(n_classes)[y]
            category_.append(category)

        category_ = np.array(category_)

        return category_

    def __call__(self, y_true, y_pred):
        if(y_true.shape[0] != y_pred.shape[0]):
            raise Exception("Label and prediction size mismatched")

        N = y_true.shape[0]
        if(len(y_true.shape) < 2):
            y_true = self.__to_categorical__(y_true)

        loss = (y_true - y_pred) ** 2
        loss = loss.mean()

        return loss
