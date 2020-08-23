import numpy as np

from dnn import NeuralNet
from layers import Dense, ReLU, SoftMax

net = NeuralNet()
net.add_layer(Dense(units=32))
net.add_layer(Dense(units=32))
net.add_layer(Dense(units=2))
net.add_layer(ReLU())
net.add_layer(SoftMax())

input_ = np.array([[1,1],[2,3]])
output_ = np.array([0,1])

net.compile(loss='mse')
net.train(input_, output_, 100)

