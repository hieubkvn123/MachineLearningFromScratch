import os
import cv2
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
from layers import Layer, Dense
from losses import Loss

### this class of neural net only works with flat vectors ###
class NeuralNet(object):
    def __init__(self, input_dim=None):
        self.input_dim = input_dim
        self.layers = list()
        self.history = {}
        self.history['error'] = [0 , 0]
        # self.history['error'][0] = 0
        # self.layers.append(Dense(input_dim))

    def add_layer(self, layer):
        if(not isinstance(layer, Layer)):
            raise Exception("argument must be an instance of layers.Layer")

        self.layers.append(layer)

    def forward(self, inputs):
        if(len(self.layers) == 0):
            raise Exception("This Neural Network is empty")

        if(inputs.shape[-1] != self.input_dim):
            raise Exception("Input dim mismatch, expecting (n,%d)" % self.input_dim)

        output = None
        for i, layer in enumerate(self.layers):
            if(i == 0):
                output = layer(inputs)
            else:
                output = layer(output)

        return output

    def __call__(self, inputs):
        outputs = self.forward(inputs)
        
        return outputs

    def compile(self, optimizer, loss):
        if(not isinstance(loss, Loss)):
            raise Exception("The loss function must be an instance of loss.Loss")

        self.optimizer = optimizer
        self.loss = loss
    
    def fit(self,x ,y, epochs=1000000):
        if(not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)):
            raise Exception("Either input or label are not numpy array")

        if(x.shape[0] != y.shape[0]):
            raise Exception("Input and label size mismatched")

        for i in range(epochs):
            output = self.forward(x)
            error = self.loss(y, output)
            self.history['error'][0] = self.history['error'][1]
            self.history['error'][1] = error

            print("[INFO] Epoch %d, Loss = %.2f" % ((i+1), error))

            for i, layer in enumerate(self.layers):
                new_thetas = self.optimizer.update_theta(self.history['error'], layer.history['weights'])
                layer.history['weights'][0] = layer.history['weights'][1]
                layer.history['weights'][1] = new_thetas
                layer.weights = new_thetas
