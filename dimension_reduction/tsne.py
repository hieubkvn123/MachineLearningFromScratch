import numpy as np 
import matplotlib.pyplot as plt 

### Loading in some sample data ### 
class_1 = np.array([[1, 2, 3], [1, 1, 2], [2, 2, 2],
                                        [2, 1, 2], [2, 2, 1], [1, 3, 1],
                                        [3, 2, 1], [3, 1, 3], [3, 3, 3]])

class_2 = np.array([[4, 2, 4], [4, 5, 3], [4, 4, 4],
                                        [5, 6, 7], [5, 5, 5], [4, 5, 4],
                                        [5, 6, 5], [6, 5, 6], [6, 6, 6]])

x = np.concatenate((class_1, class_2))
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

x_test = np.array([[1.6,2.2,2.3], [2.4,1.5,1.15], [3.4,2.4,2.5], [4.2,2.4,4.6
], [6.2,3.55,2.21], [4.2,5.4,3.5]])
y_test = np.array([1, 1, 1, -1, -1, -1])

### TSNE = T-distributed Schochastic Neighboring Embedding ###
class TSNE(object):
	def __init__(self, n_features, sigma = 1):
		### sigma is the bandwidth of gaussian kernel ###
		self.n_features = n_features
		self.sigma = sigma
		self.x = list()
		self.y = list() ### results of predictions ###
		self.N + 0
		self.feat_dim = 2

	def fit(self, x):
		### Some simple checking ###
		if(len(x.shape) < 2):
			raise Exception("Input must be an array of vectors ... ")

		self.x = x
		self.N = x.shape[0]
		self.feat_dim = x.shape[1]
		self.weights = np.ones((self.n_features, self.feat_dim))

	### First step in TSNE is to calculate (for each observation) ###
	### The probability x_i would pick x_j as its neighbor ###
	def __probability_j_given_i(self, i, j):
		'''
			i, j : the indices of the observations
		'''
		### p = exp(-1/2*sigma**2 . K(x_i, x_j))/sum(exp(-1/2*sigma**2 . K(x_i, x_k))) ###
		def kernel(x_i, x_j):
			norm2 = np.linalg.norm(x_i - x_j) ** 2
			gamma = 1/(2 * self.sigma ** 2)

			return norm2 * gamma

		numerator = np.exp(kernel(self.x[i], self.x[j]))
		denominator = 0

		for k, x_k in enumerate(self.x):
			if(k == i):
				continue

			denominator += np.exp(kernel(self.x[i], x_k))

		p_j_given_i = numerator / denominator 

		return p_j_given_i

	### Calculate p_ij (p_ij = p_ji) ###
	def __p_ij(self, i, j):
		p_j_given_i = self.__probability_j_given_i(i, j)
		p_i_given_j = self.__probability_j_given_i(j, i)

		p_ij = (p_j_given_i + p_i_given_j) / 2*self.N 

		return p_ij 

	def __q_ij(self, i, j):
		y_i = self.x[i].dot(self.weights.transpose())
		y_j = self.x[j].dot(self.weights.transpose())

		

