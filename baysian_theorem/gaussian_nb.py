import numpy as np

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

class GaussianNB(object):
	def __init__(self):
		self.x = []
		self.y = []

		self.N  = 0 
		self.feat_dim = 0

		# self.class_mean = []
		# self.class_std = []

	def get_gaussian_probability(self, x_i, mean_k_i, std_k_i):
		a = 1/(2 * std_k_i**2 * np.pi)
		b = np.exp(-1/(2 * std_k_i**2) * (x_i - mean_k_i) ** 2)

		return a * b 

	def fit(self, x, y):
		if(x.shape[0] != y.shape[0]):
			raise Exception("[INFO] Input and output must have the same length ... ")

		if(len(x.shape) < 2):
			raise Exception("[INFO] Input must be an array of vectors ...")
				
		self.x = x 
		self.y = y
		self.N = x.shape[0]
		self.feat_dim = x.shape[1]

	def get_probability_y(self, y_i):
		y = self.y[np.where(self.y == y_i)]
		num_y = y.shape[0]

		P_y = num_y / self.N

		return P_y 
		
	def get_probability_x_given_y(self, x, y):
		if(not np.isscalar(y)):
			print("[INFO] Label must be a scalar ... ")
			raise Exception("Label is not a scalar ... ")

			return None
		
		if(np.isscalar(x)):
			print("[INFO] The input must be a vector ... ")
			raise Exception("Input is a scalar")

			return None
		
		train_data = self.x[np.where(self.y == y)]
		train_mean = train_data.mean(axis=0, keepdims=True)[0]
		train_std  = train_data.std(axis=0, keepdims = True)[0]


		main_probability = 1
		probabilities = list()

		for i in range(self.feat_dim):
			probabilities.append(self.get_gaussian_probability(x[i], train_mean[i], train_std[i]))
		

		# print(train_mean)
		for probability in probabilities:
			main_probability *= probability 

		return main_probability

	def predict(self, x, probability=True):
		if(len(x.shape) < 2):
			print("[INFO] Input must be an array of vectors ... ")
			raise Exception("Invalid argument format ... ")

			return None 

		### Loop through the training dataset ###
		max_probabilities = list()
		outputs = list()

		for i in range(x.shape[0]):
			probabilities = list()

			for y_i in np.unique(self.y):
				p = self.get_probability_y(y_i) * self.get_probability_x_given_y(x[i], y_i)
				probabilities.append(p)

			class_ = np.unique(self.y)[np.argmax(probabilities)]
			max_probability = probabilities[np.argmax(probabilities)]/sum(probabilities)

			max_probabilities.append(max_probability)
			outputs.append(class_)

		if(probability):
			return outputs, max_probabilities
		else:
			return outputs

def accuracy_score(y_test, y):
	correct = 0
	for i, y_ in enumerate(y_test):
		if(y_ == y[i]):
			correct += 1

	accuracy = float(correct / y.shape[0])

	return "{0:.2f}".format(accuracy)


clf = GaussianNB()
clf.fit(x, y)

outputs, probabilities = clf.predict(x_test)

for p, y_i in zip(probabilities, outputs):
	print("Predicted class : " + str(y_i) + " | Probability = {0:.2f}".format(p))


accuracy = accuracy_score(outputs, y_test)
print("----------------------------------------------------------")
print("[INFO] Test accuracy : " + str(accuracy))
