# adaptive svm
import math
import numpy as np 
import pandas as pd 

# Note, this is for binary classification only
def kernel(kernel='linear', gamma = 0.01, r=1/2, d=2):
	'''
		gamma : 1/2*sigma**2 for the gaussian kernel
		r : the bias for polynomial kernel
		d : the power base for the polynomial kernel
	'''

	# the default linear kernel
	operation = lambda x, y : np.dot(x, y)

	if(kernel == 'rbf'):
		operation = lambda x, y : -gamma*np.linalg.norm(x-y)
	if(kernel == 'poly'):
		operation = lambda x, y : (np.dot(x,y) + r) ** d 

	return operation

# mse loss
def loss(predictions, labels):
	if(predictions.shape[0] != labels.shape[0]):
		print("[INFO] Labels and Predictions are not of the same shape ... ")
		return None
	else:
		# loop thru pairs of prediction and label
		loss = 0
		for j in range(predictions.shape[0]): # or labels.shape[0]
			loss += (predictions[j] - labels[j]) ** 2

		return math.sqrt(loss)

class_1 = np.array([[2,3], [1,2], [2.5, 3.5],
                    [1.5, 2.5], [2,2], [2.5,2.5],
                    [1.5,3],[2.5,1],[1,1]])

class_2 = np.array([[4,5],[5,5],[4,6],
                    [4.5,5.5],[5.5,5.5],[6,6],
                    [5,4],[5,6],[6,5]])

# validation data
class_1_ = np.array([[1.2,2.1],[2.1,2],[1.3,1.7]])
class_2_ = np.array([[4.4],[3.8,5.4],[4.2,3.4]])

x = np.concatenate((class_1, class_2))
y = np.array([1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

def fit(x, y, iterations=100000, alpha=0.001, l=0.01):
	'''
		- iterations is the max number of training iterations
		- alpha is the learning rate
		- l is the regularization parameter
	'''
	bias = 10.0

	if(not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)):
		print("[INFO] The input and output must be numpy array ... ")
		return None
	else:
		# check if the input is of right dimension
		if(len(x.shape) != 2 or len(y.shape) != 1):
			print("[INFO] Input and output are of wrong shape ...")
		else:
			# svm is optimized based on hinge loss
			# hinge(x,w) = Max(0, 1 - y * <x,w>)
			# we need to optimize 2/||w|| -> the regularized loss is 
			# L = summation(Max(0, 1 - y * <x,w>)) + ||w||**2/2
			K = kernel(kernel='rbf')

			# first, calculate all f_i vectors from the kernel
			dataset_size = x.shape[0]
			f_i = np.zeros((dataset_size, dataset_size))

			# construct f_i which is a matrix of kernelized inputs
			for i in range(dataset_size):
				f = np.zeros((dataset_size,))
				for j in range(dataset_size):
					f[j] = K(x[i], x[j])

				f_i[i] = f 

			# print(f_i)

			# now all we have to do is applying svm on f_1 and y
			w = np.ones((dataset_size, ), dtype=np.float32) # a vector with same size as f_i input vectors

			# loop thru the training iterations
			previous_loss = 0
			for i in range(iterations) :
				predictions = np.zeros((y.shape[0], ))
				# loop thru all the data
				for j in range(f_i.shape[0]):
					f_i_ = f_i[j]
					prediction = np.dot(w, f_i_) + bias 
					predictions[j] = prediction

					# check if this is a mis-classification case
					if(1 - y[j] * (np.dot(w, f_i_) + bias) > 0): # misclassification
						# loop thru the elements in the weight vector
						for i_ in range(w.shape[0]):
							w[i_] = w[i_] - alpha * (-y[j] * f_i_[i_] + l * w[i_])

						bias = alpha - y[j] * alpha

					else: # no misclassification occured
						# loop thru the elements of the weight vector again 
						for i_ in range(w.shape[0]):
							w[i_] = w[i_] - alpha * (l*w[i_])

				mse = loss(predictions, y)

				if(mse > previous_loss and i != 0):
					break

				previous_loss = mse
				print("[INFO] Epoch : " + str(i+1) + " | Loss = " + str(mse))

			print(w)

fit(x,y)