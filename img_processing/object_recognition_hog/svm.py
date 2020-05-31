# adaptive svm
import numpy as np 
import pandas as pd 

# Note, this is for binary classification only
def kernel(x, kernel='linear')
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
			