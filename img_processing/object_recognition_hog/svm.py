# adaptive svm
import math
import numpy as np 
import pandas as pd 

from sklearn.metrics import accuracy_score

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
		operation = lambda x, y : -gamma*(np.linalg.norm(x-y)**2)
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
class_2_ = np.array([[4,4],[3.8,5.4],[4.2,3.4]])

x = np.concatenate((class_1, class_2))
x_ = np.concatenate((class_1_, class_2_))
y = np.array([1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
y_ = np.array([1,1,1,-1,-1,-1])

# let's go a little bit object oriented shall we
class KernelSVM:
	def __init__(self):
		self.x = None 
		self.y = None 
		self.w = None
		self.kernel = None

	def fit(self,x, y, iterations=100000, alpha=0.0001, l=0.01):
		'''
			- iterations is the max number of training iterations
			- alpha is the learning rate
			- l is the regularization parameter
		'''

		# just create a copy for inference purpose
		self.x = x
		self.y = y

		bias = 10.0

		if(not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)):
			print("[INFO] The input and output must be numpy array ... ")
			return None
		else:
			# check if the input is of right dimension
			if(len(x.shape) != 2 or len(y.shape) != 1):
				print("[INFO] Input and output are of wrong shape ...")
				return None
			else:
				# svm is optimized based on hinge loss
				# hinge(x,w) = Max(0, 1 - y * <x,w>)
				# we need to optimize 2/||w|| -> the regularized loss is 
				# L = summation(Max(0, 1 - y * <x,w>)) + ||w||**2/2
				K = kernel(kernel='rbf')
				self.kernel = K

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
				self.w = np.ones((dataset_size, ), dtype=np.float32) # a vector with same size as f_i input vectors

				# loop thru the training iterations
				previous_loss = 0
				for i in range(iterations) :
					# alpha = alpha * (decay_rate * 1/(i+1))
					predictions = np.zeros((y.shape[0], ))
					# loop thru all the data
					for j in range(f_i.shape[0]):
						f_i_ = f_i[j]
						prediction = np.dot(self.w, f_i_) + bias 
						predictions[j] = prediction

						# check if this is a mis-classification case
						if(1 - y[j] * (np.dot(self.w, f_i_) + bias) > 0): # misclassification
							# loop thru the elements in the weight vector
							for i_ in range(self.w.shape[0]):
								self.w[i_] = self.w[i_] - alpha * (-y[j] * f_i_[i_] + l * self.w[i_])

							bias = bias + y[j] * alpha

						else: # no misclassification occured
							# loop thru the elements of the weight vector again 
							for i_ in range(self.w.shape[0]):
								self.w[i_] = self.w[i_] - alpha * (l*self.w[i_])

					mse = loss(predictions, y)

					if(mse > previous_loss and i > 10000):
						break

					# if the reduction in loss barely matters
					# we break the process
					if(previous_loss - mse < 1e-8 and i > 10000):
						break

					previous_loss = mse
					print("[INFO] Epoch : " + str(i+1) + " | Loss = " + "{0:.2f}".format(mse))

	def predict(self, x):
		predictions = []

		# for each of the new data
		for i in range(x.shape[0]):
			# loop thru the training dataset
			prediction = 0
			for j in range(self.x.shape[0]):
				prediction += self.y[j] * self.w[j] * self.kernel(x[i], self.x[j])

			if(prediction > 0):
				predictions.append(1)
			else:
				predictions.append(-1)

		return predictions 


svm = KernelSVM()
svm.fit(x,y)

predictions = svm.predict(x_)
# print(predictions)

# time to make prediction to see if it is true
print("[INFO] Accuracy = " + str(accuracy_score(predictions, y_)))