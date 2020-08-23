import numpy as np 

class Loss(object):
	def __init__(self):
		pass

class MSE(Loss):
	def __init__(self):
		pass 

	def __call__(self, y_true, y_pred):
		if(not isinstance(y_true, np.ndarray)):
			raise Exception("Label must be a numpy array")

		if(len(y_pred.shape) < 2):
			raise Exception("Predictions must be a numpy array")

		loss = (y_true - y_pred) ** 2
		loss = np.mean(loss)# .sum()

		return loss