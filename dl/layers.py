import numpy as np 

class Layer(object):
	def __init__(self):
		pass

class Activation(Layer):
	def __init__(self):
		pass 

class DenseLayer(Layer):
	def __init__(self):
		pass

class ReLU(Activation):
	def __init__(self):
		pass 

	def __call__(self, inputs):
		if(not isinstance(inputs, np.ndarray)):
			raise Exception("Input must be a numpy array")

		if(len(inputs.shape) < 2):
			raise Exception("Input must be a numpy array of arrays")

		inputs[np.where(inputs < 0)] = 0

		return inputs

class SoftMax(Activation):
	def __init__(self):
		pass

	def __call__(self, inputs):
		if(not isinstance(inputs, np.ndarray)):
			raise Exception("Input must be a numpy array")

		if(len(inputs.shape) < 2):
			raise Exception("Input must be a numpy array of arrays")

		outputs = np.exp(inputs) / np.exp(inputs).sum(axis=1, keepdims=True)

		return outputs

class Dense(DenseLayer):
	def __init__(self, units = None):
		if(units is None):
			raise Exception("Unit must not be none")

		self.input_shape = None
		self.weights = None
		self.inputs = None 
		self.units = units

	def __call__(self, inputs):
		if(not isinstance(inputs, np.ndarray)):
			raise Exception("Input must be a numpy array")

		if(len(inputs.shape) < 2):
			raise Exception("Input must be a numpy array of arrays")

		### if first time called ###
		if(self.weights is None):
			batch_size = inputs.shape[0]
			self.input_shape = inputs.shape[1]
			self.weights = np.ones((self.input_shape, self.units)) / batch_size

		### Save inputs for back probagation ###
		self.inputs = inputs
		outputs = inputs @ self.weights
		return outputs

	def backward(self, lr, error):
		for i in range(self.input_shape):
			gradient = lr * error * self.inputs[:,i]
			gradient = gradient.sum()

			self.weights[i] -= gradient


