import os
import sys
import traceback
import cv2
import numpy as np

class Layer(object):
    def __init__(self):
        pass
       
class Activation(Layer):
    def __init__(self, name='none'):
        self.name = name

    def softmax(self, inputs):        
        if(not isinstance(inputs, np.ndarray)):
            raise Exception("Input must be a numpy array")
        
        if(len(inputs.shape) < 2):
            raise Exception("Input must be a 2D array")

        sum_exp = sum(list(map(lambda x_ : np.exp(x_), inputs))) 
        x = np.exp(inputs) / sum_exp
        
        return x

    def __call__(self, inputs):
        ### check if the input is 2D ###
        if(not isinstance(inputs, np.ndarray)):
            raise Exception("Input must be a numpy array")
        
        if(len(inputs.shape) < 2):
            raise Exception("Input must be a 2D array")

        activation = None
        if(self.name == 'relu'):
            activation = lambda x : np.clip(x, 1e-8, 1e8)

        elif(self.name == 'tanh'):
            activation = lambda x : np.tanh(x)
        
        elif(self.name == 'sigmoid'):
            activation = lambda x : 1/(1 + np.exp(-x))

        elif(self.name == 'softmax'):
            activation = lambda x : self.softmax(x)

        elif(self.name == 'none'):
            activation = lambda x : x

        else:
            raise Exception("Invalid activation function")

        return activation(inputs)

class Dense(Layer):
    def __init__(self, units=None, activation='relu'):
        super(Dense, self).__init__()
        self.units = units
        self.weights = None
        self.activation = Activation(activation)

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
            self.weights = np.ones((self.input_dim, self.units))

            ### Keep history of weights and errors for gradients ###
            self.history = {}
            self.history['weights'] = [[], []]
            self.history['weights'][0] = np.zeros((self.input_dim, self.units)) 
            self.history['weights'][1] = self.weights
            # self.history['error'] = 0

        output = inputs @ self.weights
        output = self.activation(output)

        return output

    def update(self):
        pass

    def __call__(self, inputs):
        outputs = self.forward(inputs)

        return outputs
