import numpy as np
import matplotlib.pyplot as plt ### For result visualization ###

# this is naive bayes implementation
class_1 = np.array([[1, 2, 3], [1, 1, 2], [2, 2, 2],
					[2, 1, 2], [2, 2, 1], [1, 3, 1],
					[3, 2, 1], [3, 1, 3], [3, 3, 3]])

class_2 = np.array([[4, 2, 4], [4, 5, 3], [4, 4, 4],
					[5, 6, 7], [5, 5, 5], [4, 5, 4],
					[5, 6, 5], [6, 5, 6], [6, 6, 6]])

x = np.concatenate((class_1, class_2))
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

x_test = np.array([[1.6,2.2,2.3], [2.4,1.5,1.15], [3.4,2.4,2.5], [4.2,2.4,4.6], [6.2,3.55,2.21], [4.2,5.4,3.5]])
y_test = np.array([1, 1, 1, -1, -1, -1])


### To declare private methods in python ###
### prefix them with double underscore "__" ###
class KMeans(object):
	def __init__(self, n_clusters = 2):
		self.n_clusters = n_clusters
		self.N = 0
		self.feat_dim = 2 ### by default ###

		self.x = list()
		self.y = list() ### empty ###

	def fit(self, x):
		### Some basic checking ###
		if(len(x.shape) < 2):
			raise Exception("Input must be an array of vectors ... ")


		self.x = x
		self.N = x.shape[0]
		self.feat_dim = x.shape[1]

		self.centroids = np.zeros((self.n_clusters, self.feat_dim))

		self.__initCluster()
		self.__train()

		return self.y

	### Generate the centroids for k clusters ###
	def __initCluster(self):
		### Randomly choose k observations in the dataset ###
		for i in range(self.n_clusters):
			### Choose an observation from N observations ###
			self.centroids[i] = self.x[np.random.randint(0, self.N)]

	### distance matrix 
	def __dist_mat(self, x):
		### Calculate the euclidean distance from the ###
		### observation to each of the centroid ###
		distance_matrix = np.zeros((self.n_clusters,))

		for i, centroid in enumerate(self.centroids):
			distance = np.linalg.norm(x - centroid)
			distance_matrix[i] = distance

		return distance_matrix


	def __train(self, iterations = 100): 
		### For each observation in the dataset ###
		### Check what centroid is the closest to the data point ###
		### termination condition : if the labels does not change ###
		previous_labels = np.zeros((self.N, ))
		self.y = np.zeros((self.N,))

		for i_ in range(iterations):
			previous_labels = self.y.copy()
			for i in range(self.N):
				### here we are gonna figure out each datapoint belong to which class ###
				### by getting the min distance to centroids ###
				distance_matrix = self.__dist_mat(self.x[i])
				label = np.argmin(distance_matrix)
				self.y[i] = label

			if((previous_labels == self.y).all()):
				### If we are not at the first iteration ###
				if(i_ > 0):
					print("[INFO] KMeans converged at iteration : %d" % (i_ + 1))
					break ### break training process ### 

			### now update the centroids ###
			for cluster in range(self.n_clusters):
				### get the data in the present cluster ###
				data = self.x[np.where(self.y == cluster)]

				### update the centroids ### 
				self.centroids[cluster] = data.mean(axis=0, keepdims=True)

kmeans = KMeans(n_clusters = 2)
labels = kmeans.fit(x)

axes = plt.axes(projection='3d')
### Visualizing the results ###
for cluster in np.unique(labels):
	data = x[np.where(labels == cluster)]

	axes.scatter3D(data[:,0], data[:,1], data[:,2], alpha=0.8)

plt.show()