import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE as SklearnTSNE

### Loading in some sample data ### 
class_1 = np.array([[1, 2, 3], [1, 1, 2], [2, 2, 2],
										[2, 1, 2], [2, 2, 1], [1, 3, 1],
										[3, 2, 1], [3, 1, 3], [3, 3, 3]])

class_2 = np.array([[4, 2, 4], [4, 5, 3], [4, 4, 4],
										[5, 6, 7], [5, 5, 5], [4, 5, 4],
										[5, 6, 5], [6, 5, 6], [6, 6, 6]])

x = np.concatenate((class_1, class_2))

### TSNE = T-distributed Schochastic Neighboring Embedding ###
class TSNE(object):
	def __init__(self, n_features, sigma = 5):
		### sigma is the bandwidth of gaussian kernel ###
		self.n_features = n_features
		self.sigma = sigma
		self.x = list()
		self.y = list() ### results of predictions ###
		self.N = 0

	def fit(self, x):
		### Some simple checking ###
		if(len(x.shape) < 2):
			raise Exception("Input must be an array of vectors ... ")

		self.x = x
		self.N = x.shape[0]

		### initialize y because we are going to optimize ###
		### the kullback leiber divergence with respect to y_i ###
		self.y = np.array(np.random.rand(self.N, self.n_features))
		self.__train(iterations=100, alpha = 0.1)

		return self.y
	### First step in TSNE is to calculate (for each observation) ###
	### The probability x_i would pick x_j as its neighbor ###
	def __probability_j_given_i(self, i, j):
		'''
			i, j : the indices of the observations
		'''
		### p = exp(-1/2*sigma**2 . K(x_i, x_j))/sum(exp(-1/2*sigma**2 . K(x_i, x_k))) ###
		def kernel(x_i, x_j):
			norm2 = -np.linalg.norm(x_i - x_j) ** 2
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

	### Calculate q_ij -> the P after transformation ###
	### optimize weights using the difference between 2 distributions ###
	### we are going to use Kullback Leiber divergence (entropy) ###
	def __q_ij(self, i, j):
		y_i = self.y[i]
		y_j = self.y[j]

		def kernel(y_i, y_j):
			norm2 = np.linalg.norm(y_i - y_j) ** 2
			return (1 + norm2) ** (-1)

		numerator = kernel(y_i, y_j)
		denominator = 0

		for k, y_k in enumerate(self.y):
			if(k == i):
				continue

			denominator += kernel(y_i, y_k)

		return numerator / denominator

	### Now the training phase ###
	def __train(self, iterations = 10, alpha = 200):
		### gonna follow basic gradient descent ###
		for i_ in range(iterations):
			kl = 0
			for i in range(self.x.shape[0]):
				### Calculating the gradient of kullback leiber divergence ###
				### With respect to y_i => we are optimizing y_i ###
				sum_value = 0
				for j in range(self.x.shape[0]):
					if(i == j):
						continue

					p_ij = self.__p_ij(i, j)
					q_ij = self.__q_ij(i, j)

					kl += p_ij * np.log(p_ij / q_ij)
					sum_value += 4*(p_ij - q_ij) * (self.y[i] - self.y[j]) * ((1 + np.linalg.norm(self.y[i] - self.y[j]) ** 2 )**(-1))

			
				self.y[i] -= alpha * sum_value 
			
			print("[INFO] Iteration %d, KL = %.2f" % ((i_ + 1), kl))

tsne = TSNE(n_features = 2)
results = tsne.fit(x)

sklearn_tsne = SklearnTSNE(n_components = 2, perplexity=5)
results_sk = sklearn_tsne.fit_transform(x)

class_1 = results[:9]
class_2 = results[9:]

print(class_1, class_2)

class_1_sk = results_sk[:9]
class_2_sk = results_sk[9:]

print(class_1_sk, class_2_sk)

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].scatter(class_1[:, 0], class_1[:, 1], color ='green')
ax[0].scatter(class_2[:, 0], class_2[:, 1], color = 'red')

ax[1].scatter(class_1_sk[:, 0], class_1_sk[:, 1], color ='green')
ax[1].scatter(class_2_sk[:, 0], class_2_sk[:, 1], color = 'red')

plt.show()
