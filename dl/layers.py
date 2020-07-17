import os
import sys
import traceback
import cv2
import numpy as np

class Layer(object):
    def __init__(self):
        pass
        
class Dense(Layer):
    def __init__(self, units=None):
        super(Dense, self).__init__()
        self.units = units
        self.weights = None

    ### the input shape should be [num_data, input_shape]
    def forward(self, inputs):
        ### initialize your weights as you have the input ###
        ### the weights (theta) shape should be  ###
        if( not isinstance(inputs, np.ndarray)):
            raise Exception("The input is not an array")

        if(len(inputs.shape) < 2):
            raise Exception("Input dimension is insufficient, expecting a 2D array")

        self.input_dim = inputs.shape[-1] # the last dimension
        
        ### if the weights has not been initialized ###
        if(self.weights is None):
            self.weights = np.random.rand(self.input_dim, self.units)

            ### Keep history of weights and errors for gradients ###
            self.history = {}
            self.history['weights'] = [[], []]
            self.history['weights'][0] = np.zeros((self.input_dim, self.units)) 
            self.history['weights'][1] = self.weights
            # self.history['error'] = 0

        output = inputs @ self.weights

        return output

    def update(self):
        pass

    def __call__(self, inputs):
        outputs = self.forward(inputs)

        return outputs
