# adaptive svm
import numpy as np 
import pandas as pd 

# Note, this is for binary classification only
def kernel(x, kernel='linear', gamma = 0.01):
	operation = lambda x, y : np.dot(x, y)

	if(kernel == 'rbf'):
		operation = lambda x, y : -gamma*np.linalg.norm(x-y)
	if(kernel == 'poly'):
		operation = lambda x, y : 
def fit(x, y):
	if(not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)):
		print("[INFO] The input and output must be numpy array ... ")
		return None
	else:
		# check if the input is of right dimension
		if(len(x.shape) != 2 or len(y.shape) != 1):
			print("[INFO] Input and output are of wrong shape ...")
		else:
			# svm is optimized based on hinge loss
			# hinge(x,w) = Max(0, y - <x,w>)
			# we need to optimize 2/||w|| -> the regularized loss is 
			# L = summation(Max(0, y - <x,w>)) + ||w||/2
			