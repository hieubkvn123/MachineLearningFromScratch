import os
import numpy as np

class Optimizer(object):
    def __init__(self):
        pass 
    
    def __call__(self):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def __call__(self):
        pass

    def update_theta(self, error, thetas):
        gradient = self.__compute_gradient__(error, thetas)
        new_theta = thetas[1] - self.learning_rate * gradient

        return new_theta

    def __compute_gradient__(self, error, thetas):
        error_t_0 = error[0]
        error_t_1 = error[1]

        # print(error)

        theta_t_0 = thetas[0]
        theta_t_1 = thetas[1]

        #print(error_t_0, error_t_1, theta_t_0, theta_t_1)
        gradient = (error_t_1 - error_t_0) / (theta_t_1 - theta_t_0)
        gradient = np.clip(gradient, 1e-8, 1e8)
        # print(error_t_0, error_t_1)
        gradient = gradient.mean(axis=0, keepdims=True)
        # print(gradient)

        return gradient
