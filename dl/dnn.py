import numpy as np 

from layers import Layer, Activation, DenseLayer
from losses import Loss, MSE 

LOSS_LIST = ['mse', 'crossentropy']

class NeuralNet(object):
	def __init__(self):
		self.layers = list()
		self.loss = None
		self.lr = None
		self.units = None

	def add_layer(self, layer):
		if(not isinstance(layer, Layer)):
			raise Exception("Invalid layer")

		if(isinstance(layer, DenseLayer)):
			self.units = layer.units

		self.layers.append(layer)

	def __call__(self, inputs):
		if(len(self.layers) < 1):
			raise Exception("Neural Network is empty")

		if(not isinstance(inputs, np.ndarray)):
			raise Exception("Input must be a numpy array")

		if(len(inputs.shape) < 2):
			raise Exception("Input must be a numpy array of arrays")			

		output = self.layers[0](inputs)

		for i in range(1, len(self.layers)):
			output = self.layers[i](output)

		return output

	def compile(self, loss='mse', lr=0.001):
		global LOSS_LIST

		if(loss not in LOSS_LIST and not isinstance(loss, Loss)):
			raise Exception("Invalid loss")

		if(isinstance(loss, str)):
			if(loss == 'mse'):
				self.loss = MSE()

		else:
			self.loss = loss 

		self.lr = lr 

	def train(self, inputs, labels, epochs):
		if(not isinstance(labels, np.ndarray)):
			raise Exception("Label must be a numpy array")

		### if labels are scalars ###
		if(len(labels.shape) == 1):
			labels_ = list()
			for label in labels:
				new_label = np.eye(self.units)[label]
				labels_.append(new_label)
			labels = np.array(labels_)

		for i in range(epochs):
			output = self.__call__(inputs)
			print(output)
			loss = self.loss(output, labels)
			print(loss)

			for layer in self.layers:
				if(isinstance(layer, DenseLayer)):
					layer.backward(self.lr, loss)
					# print(layer.weights)
