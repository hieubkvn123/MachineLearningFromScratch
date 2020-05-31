# this is another version of PCA (not the one at root directory)
# this one will be more adaptive to work with vectors in any vector space
# whether R(2) or R(n)
import numpy as np
import pandas as pd

# default will be 2 components reduction
def pca(x, n_components=2):
	if(isinstance(x, np.ndarray)):
		if(len(x.shape) != 2):
			# by now just default that the input is a matrix
			print("[INFO] Invalid matrix shape ... ")
			return None
		else: 
			# first, calculate the covariance matrix
			covariance = np.cov(x.transpose())

			# then just calculate the eigen values and vectors
			eig_vals, eig_vecs = np.linalg.eig(covariance)

			# get top eigen values
			top_eig_vals = eig_vals.argsort()[::-1]

			# get top eigen vectors
			top_eig_vecs = eig_vecs[top_eig_vals]
			top_eig_vecs = top_eig_vecs[0:n_components]

			y = np.dot(x, top_eig_vecs.transpose())

			return y
	else:
		print("[INFO] The input matrix must be a numpy ndarray ... ")
		return None

# okey, that's about it